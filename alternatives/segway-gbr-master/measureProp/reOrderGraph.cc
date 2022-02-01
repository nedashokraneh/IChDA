

/////////////////////////////////////////////////////////////////////
// contains all the helper functions to reorder the graph so to 
// improve cache performance.
// 
//

///////////////////////////////////////////////////////////////////////
// to compute the cardinality of the intersection set 
// between two sorted lists
// NOTE THAT IS IT ASSUMED THAT THE LISTS ARE SORTED
// IN DESCENDING ORDER. 
// currently defined inline for speed. 
//
//
inline 
unsigned short intersect(const unsigned int *a, 
			 const unsigned int *b, 
			 const unsigned short len1, 
			 const unsigned short len2) {

  unsigned short n = 0, nPtrA = 0, nPtrB = 0;

  while (nPtrA < len1 && nPtrB < len2) {

    if (a[nPtrA] == b[nPtrB]) {
      n++;
      nPtrA++; 
      nPtrB++;
    }
    else if ( a[nPtrA] < b[nPtrB] ) {
      nPtrB++;
    }
    else {
      nPtrA++;
    }

  }

  return(n);

}


///////////////////////////////////////////////////////////////////////
// a helper function for sorting according to frame_ids.
//
//
bool 
sortfunc (const NNIdxWeight &t1, const NNIdxWeight &t2) {
  return (t1.idx > t2.idx);
}


//////////////////////////////////////////////////////////////////
// to sort the neighors of each node in descending order
// of their indices. 
//
//
//
void 
sortIndividualNodes(node *graph, 
		    unsigned int numNodesInGraph) {


  for (unsigned int i = 0; i < numNodesInGraph; i++) {

    // define for fast access
    const node* curNode = &graph[i];
    
    // a data strucutre to aid the sort. 
    NNIdxWeight *tmp = new  NNIdxWeight[curNode->NN];

    for (unsigned int j = 0; j < curNode->NN; j++) {
      tmp[j].idx = curNode->idx[j];
      tmp[j].weight = curNode->weight[j];
    }

    sort(tmp, tmp + curNode->NN, sortfunc);

    for (unsigned int j = 0; j < curNode->NN; j++) {
      curNode->idx[j] = tmp[j].idx;
      curNode->weight[j] = tmp[j].weight;

    }

  }

}


//////////////////////////////////////////////////////////////////
// to map the NN indices to the new base.
//
//
//
void reMapIdx(node *graph, unsigned int *mapping, 
	      unsigned int numNodesInGraph){

  printf("Re-mapping indices....\n");
 
  for (unsigned int i = 0; i < numNodesInGraph; i++) {
   
    const node *tmp = &graph[i];

    for (unsigned int j = 0; j < tmp->NN; j++) 
      tmp->idx[j] = mapping[tmp->idx[j]];

  }

}

//////////////////////////////////////////////////////////////////////
// graph re-ordering function. 
//
//
//
node* reOrderGraph(node *graph, 
		  const unsigned int numNodesInGraph) {

  // variables for book-keeping. 
  int iCard, bestICard, bestIdx, max, max_pos; 

  // sort the individual nodes
  sortIndividualNodes(graph, numNodesInGraph);

  // define an array that holds the mapping from position in graph
  // to position in newGraph.
  unsigned int *oldToNew = new unsigned int[numNodesInGraph];

  // this goes from new to old. 
  unsigned int *newToOld = new unsigned int[numNodesInGraph];

  // an array that indicates whether node i has been added to the 
  // new graph. note that i here is the index in the old graph. 
  // a node CANNOT be added twice !!
  bool *added = new bool[numNodesInGraph];  
  for (unsigned int i = 0; i < numNodesInGraph; i++) 
    added[i] = false;

  // define the newGraph. 
  node *newGraph = new node[numNodesInGraph];
  unsigned int numAdded = 0;

  // add the first node. we choose this to be the first node
  // in the graph, but this can be any node. 
  added[0] = true;
  oldToNew[0] = numAdded;
  newToOld[numAdded] = 0;
  newGraph[numAdded++] = graph[0];

  // do until all the nodes have been added. 
  while ( numAdded < numNodesInGraph) {

    // first find the node in the original graph that is closest to
    // node that was last added to the graph. 
    const node *curNode = &graph[newToOld[numAdded - 1]];

    bestICard = -1, bestIdx = -1;

    // step through each neighbor of curNode. 
    for (unsigned int j = 0; j < curNode->NN; j++) {

      const node *curNeighbor = &graph[curNode->idx[j]];

      max = -1, max_pos = -1;

      // step through each of the neighors' neighbor of node i.
      for (unsigned short k = 0; k < curNeighbor->NN; k++) {

	const node *curNeighborsNeighbor = &graph[curNeighbor->idx[k]];
	
	if (!added[curNeighbor->idx[k]])
	  iCard = intersect(curNode->idx, curNeighborsNeighbor->idx, 
			    curNode->NN, curNeighborsNeighbor->NN);
	else 
	  iCard = 0;

	if (iCard > max)  {
	  max = iCard;
	  max_pos = k;
	}

      } // end of k 

      if (max > bestICard) {
	bestICard = max;
	bestIdx = curNeighbor->idx[max_pos];
      }

    } // end of j

    //    printf("bestICard = %d, bestIdx = %d, numAdded = %d\n", 
    //	   bestICard, bestIdx, numAdded);

    // if the cardinality of the best intersection is zero, then
    // simply add just any other node from list of unadded nodes. 
    if (bestICard <= 0)  { 

      unsigned int count = 0;
      while (added[count] && count < numNodesInGraph) { count++; } 
      bestIdx = count;

    }

    // two sanity checks.
    // at this point added[bestIdx] must be false, else something is amiss. 
    /*assert (! added[bestIdx] );
    // to avoid a potential seg-fault;
    assert(bestIdx <= numNodesInGraph);*/

    added[bestIdx] = true;
    oldToNew[bestIdx] = numAdded;
    newToOld[numAdded] = bestIdx;
    newGraph[numAdded++] = graph[bestIdx];

    if (numAdded % 100000 == 0)
      printf("Done with %d of %d\n", numAdded, numNodesInGraph);
    
  } // end of while loop

  // need to change all NN idx's vis-a-vis oldToNew
  reMapIdx(newGraph, oldToNew, numNodesInGraph);

  // clean-up after yourself. 
  delete[] added;
  delete[] oldToNew;
  delete[] newToOld;

  return(newGraph);
  
}



//////////////////////////////////////////////////////////////////////
// to compute the average cardinality of the intesection between
// nodes i and i + 1 in a given graph
//
//
float computeAverageIntersectionCard(node *graph, 
				     const unsigned int numNodesInGraph) {

  // sort the individual nodes
  sortIndividualNodes(graph, numNodesInGraph);
  
  float sum = 0;

  for (unsigned int i = 0; i < numNodesInGraph - 1; i++) 
    sum += intersect(graph[i].idx, graph[i + 1].idx, 
		     graph[i].NN, graph[i + 1].NN);

  sum /= numNodesInGraph;

  return(sum);

}
