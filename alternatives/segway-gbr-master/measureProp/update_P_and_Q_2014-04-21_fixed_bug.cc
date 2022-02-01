////////////////////////////////////////////////////////////////
// contains the "inner-most" loop. The code to update the
// distribution P and Q. 
//
//

////////////////////////////////////////////////////////
//
// to send the arguments for the thread in 
// the form a struct.
//
struct thread_data {
  int index;
  unsigned short numThreads;
  unsigned int numNodesInGraph;
  unsigned short numClasses;
  float mu;
  float nu;
  float conv;
  node *graph;
  bool useLP;
};


///////////////////////////////////////////////////////////////////
// to update P (Q is fixed)
// see paper for detailed equations. 
//
//
void *update_P(void *ptr) {

  thread_data *index;
  index = (thread_data*) ptr;

  //fprintf(stderr, "thread %d\t", index->index);
  //fflush(stderr);

  // load into registers to avoid aliasing.
  const unsigned short l_numThreads = index->numThreads;
  const unsigned int l_numNodesInGraph = index->numNodesInGraph;
  const unsigned short l_numClasses = index->numClasses;
  const float mu = index->mu;
  const float l_nu = index->nu;
  node *graph = index->graph;

  // see the equations.
  float *beta = new float[l_numClasses];

  // iterate over all the nodes
  /*  for (unsigned int i = lowIndex[index->index]; 
      i <= highIndex[index->index]; i++) {*/

   // to avoid lookups
  node *tmp = &(graph[index->index]);

  for (unsigned int i = index->index; i < l_numNodesInGraph; i+= l_numThreads, tmp += l_numThreads) {


    // XXX print pDist  
    float *myp = tmp->pDist();
    printf("(P) %d -- Before:", i);
    for (unsigned int j = 0; j < l_numClasses; j++, myp++)  {
      printf("\t%f", *myp);
    }


    // first compute the sum of weights for this node -- this is 
    // potentially one place where things can be speed-up by using
    // more memory. 
    float sum_of_weights = config.selfWeight * mu;
    for (unsigned int k = 0; k < tmp->NN ; k++) {
        sum_of_weights += tmp->weight[k] * mu;
    }

    // this is the same for all the classes.
    float alpha = l_nu + sum_of_weights;
    printf(" (alpha: %f)", alpha);

    // compute this for each class.
    float den = 0.0;

    // iterate over each class.
    for (unsigned int j = 0; j < l_numClasses; j++) {
        // i and j define a particular element of PDist.
        float cur_beta = 0.0;

        // for each NN
        for (short k = 0; k < tmp->NN; k++) {
#if USE_FAST_LOG
            cur_beta += tmp->weight[k] * mu * (lut_log( *( graph[ tmp->idx[k] ].qDist() + j) + SMALL, mytable, TABEXP ) - 1.0 );
#else
            cur_beta += tmp->weight[k] * mu * (log( *( graph[ tmp->idx[k] ].qDist() + j) + SMALL) - 1.0 ); 
#endif
            printf(" (cur_beta 1: %f) ", cur_beta);
        }

        // add the self-weight component.
#if USE_FAST_LOG
        cur_beta += config.selfWeight * mu * ( lut_log( *(tmp->qDist() + j) + SMALL, mytable, TABEXP ) - 1.0 );
#else
        cur_beta += config.selfWeight * mu * ( log( *(tmp->qDist() + j) + SMALL) - 1.0 ); 
#endif
        printf(" (cur_beta 2: %f) ", cur_beta);

        // Complete numerator expression
        cur_beta = exp( ( cur_beta - l_nu ) / alpha );
        printf(" (cur_beta 3: %f) ", cur_beta);

        // store for normalization.
        den += cur_beta;

        // store result
        beta[j] = cur_beta;
    }

    // sanity check
    assert(den != 0.0);


    // update the parameters for each class.    
    printf("\t After:");
    float *p = tmp->pDist();
    for (unsigned int j = 0; j < l_numClasses; j++, p++)  {
      printf("\t%f", beta[j] / den);
      *p = beta[j] / den;
    }
    printf("\n");

  }

  // free up memory. 
  delete [] beta;
  return(NULL);  
}

/////////////////////////////////////////////////////////////////////
// to update Q holding P fixed. 
//
//
//
void *update_Q(void *ptr) {

  thread_data *index;
  index = (thread_data*) ptr;

  float max = 0.0, tmp_num = 0.0, sum_of_weights = 0.0,
    convCrit = 0.0;


  // load into registers to avoid aliasing.
  const unsigned short l_numThreads = index->numThreads;
  const unsigned int l_numNodesInGraph = index->numNodesInGraph;
  const unsigned short l_numClasses = index->numClasses;
  const float mu = index->mu;
  node *graph = index->graph;

  //fprintf(stderr, "thread %d mu %f\t", index->index, mu);
  //fflush(stderr);

  node *tmp = &(graph[index->index]);
  
  for (unsigned int i = index->index; i < l_numNodesInGraph; 
       i+= l_numThreads, tmp += l_numThreads) {

    // get pointer to current qDist.
    float *curQDist = tmp->qDist();

    // XXX print pDist  
    float *myq = tmp->qDist();
    printf("(Q) %d -- Before:", i);
    for (unsigned int j = 0; j < l_numClasses; j++, myq++)  {
      printf("\t%f", *myq);
    }

    // for convergence criterion
    max = 0.0;
    
    // for each class "f = j".
    for (unsigned int j = 0; j < l_numClasses; j++, curQDist++) {

      float num = 0.0, den = 0.0;

      // if the vertex is labeled
      if (tmp->labeled == 'l')  {
        if (!config.measureLabels) {
          // increment numerator if label equals j
          if ( j == tmp->zeroEntropyLabel )
            num++;
        } else {
          // add the label probability to numerator
          num += tmp->labelDistPtr[j];
        }

        // add one to denominator irrespective
        den++;
      }

      // step through all the neigbors
      tmp_num = 0.0, sum_of_weights = 0.0;
      
      for (unsigned short k = 0; k < tmp->NN; k++)  {
        tmp_num += tmp->weight[k] * mu * *( graph[ tmp->idx[k] ].pDist() + j);
        sum_of_weights += tmp->weight[k] * mu;
      }
      tmp_num +=  config.selfWeight * mu * *(tmp->pDist() + j);
      sum_of_weights += config.selfWeight * mu;
      
      // update numerator and denominator.
      num += tmp_num;
      den += sum_of_weights;

      // sanity check
      if (den == 0.0) {
	printf("Sum of weights is zero. This happens if you have a graph vertex");
	printf(" with no neighbors\n");
        printf("This is a problem, please give me another graph");
	GError("", 1);
      }

      // update the probability
      float oldVal = *curQDist, newVal = num/den,
	change = newVal/ (oldVal + SMALL);
      max = (change > max)?change:max;
      
      *curQDist = newVal;
    
    }

    // XXX print qDist  
    myq = tmp->qDist();
    printf("\tAfter:");
    for (unsigned int j = 0; j < l_numClasses; j++, myq++)  {
      printf("\t%f", *myq);
    }
    printf("\n");

    convCrit += ( ( (tmp->labeled == 'l') ? 1 : 0 ) 
		  + sum_of_weights )*log(max + SMALL);

    //    if ( i % 500000 == 0 && i > 0)
    // printf("\t\tDone with %d vertices on thread %d\n", i, index->index);

  }

  index->conv = convCrit;
  
  return(NULL);
}


///////////////////////////////////////////////////////////////////
// to update P for the squared loss case (Q is fixed)
// note that this also implements LP (label propagation)
// see paper for detailed equations.
//
//
void *update_P_SQL(void *ptr) {


  thread_data *index;
  index = (thread_data*) ptr;

  fprintf(stderr, "updating P in thread %d\n", index->index);
  fflush(stderr);

  const unsigned short numThreads = index->numThreads;
  const unsigned int numNodesInGraph = index->numNodesInGraph;
  const unsigned short numClasses = index->numClasses;
  node *graph = index->graph;


  //  cout << "updating P ... " << endl;
  for (unsigned int i = index->index; i < numNodesInGraph;
       i+= numThreads) {

    // to avoid lookups
    node *tmp = &graph[i];

    // update the parameters for each class -- in this case a simple copy
    float *p = tmp->pDist();
    float *q = tmp->qDist();
    for (unsigned int j = 0; j < numClasses; 
	 j++, p++, q++)
      *p = *q;

    //    if ( i % 500000 == 0 && i > 0)
    //      printf("\t\tDone with %d vertices\n", i);

  }

  return(NULL);

}


/////////////////////////////////////////////////////////////////////
// to update Q holding P fixed for the squared loss case.
//
//
//
void *update_Q_SQL(void *ptr) {

  thread_data *index;
  index = (thread_data*) ptr;

  fprintf(stderr, "updating Q in thread %d\n", index->index);
  fflush(stderr);

  const unsigned short numThreads = index->numThreads;
  const unsigned int numNodesInGraph = index->numNodesInGraph;
  const unsigned short numClasses = index->numClasses;
  const float mu = index->mu;
  const float nu = index->nu;
  node *graph = index->graph;
  const bool useLP = index->useLP;

  //  cout << "updating Q ... " << endl;
  for (unsigned int i = index->index; i < numNodesInGraph;
       i+= numThreads) {

    node *tmp = &graph[i];

    // for each class "f = j".
    for (unsigned int j = 0; j < numClasses; j++) {

      float num = 0.0, den = 0.0;

      // if the vertex is labeled
      if (tmp->labeled == 'l')  {

        if (!config.measureLabels) {
          // increment numerator if label equals j
          if ( j == tmp->zeroEntropyLabel )
            num++;
        } else {
          // add the label probability to numerator
          num += tmp->labelDistPtr[j];
        }

        // add one to denominator irrespective
        den++;
      }

      // step through all the neigbors
      float tmp_num = 0.0, sum_of_weights = 0.0;

      for (unsigned short k = 0; k < tmp->NN; k++)  {

        tmp_num += tmp->weight[k] * mu * *( graph[ tmp->idx[k] ].pDist() + j);
        sum_of_weights += tmp->weight[k] * mu;

      }

      // update numerator and denominator.
      num += tmp_num;
      den += sum_of_weights;

      // add the "entropy" term
      num += nu/numClasses;
      den += nu;

      // sanity check
      if (den == 0.0) {
	
        fprintf(stderr, "node = %d, sum_of_wieights = %f, NN = %d, mu = %f\n", 
		i, sum_of_weights, tmp->NN, mu);
        GError("something wrong with the graph, quitting while updating qDist", 1);

      }

      // update the probability
      *(tmp->qDist() + j) = num/den;

    } // end of iteration over each element of q for a given vertex.


    // now reset the labeled guys back to original value.
    // note that needs to be done only in the case of LP.
    if (useLP && graph[i].labeled == 'l') {
        fprintf(stderr, "XXX resetting labeled values!");

      for (unsigned int k = 0; k < numClasses; k++) {
        if (!config.measureLabels) {
          if ( k == graph[i].zeroEntropyLabel )
            *(graph[i].qDist() + k) = 1.0;
          else
            *(graph[i].qDist() + k) = SMALL;
        } else {
          *(graph[i].qDist() + k) = graph[i].labelDistPtr[k];
        }
      }

    }

    //    if ( i % 500000 == 0 && i > 0)
    //      printf("\t\tDone with %d vertices\n", i);

  }

  return(NULL);

}

