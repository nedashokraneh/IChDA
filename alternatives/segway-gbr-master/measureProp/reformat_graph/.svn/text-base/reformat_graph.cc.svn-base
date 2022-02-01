////////////////////////////////////////////////////////////////////
//
// To reformat the graph. This takes a graph in the following format:
//
// <number_of_vertices (unsigned int)>
// <file id (unsigned int)> <frame id (unsigned int)>
// <numNeighbors (unsigned short)>
// <file id of neighbor 1 (unsigned int)> <frame id of neighbor 1 (unsigned int)>
// <weight for neighbor 1 (float)>
// <file id of neighbor 2 (unsigned int)> <frame id of neighbor 2 (unsigned int)>
// <weight for neighbor 2 (float)>
// ..... 
//
// and converts it to the following format:
//
// <number_of_vertices (unsigned int)>
// <vertex id (unsigned int)> <numNeighbors (unsigned short)>
// <index of neighbor 1 (unsigned int)> <weight for neighbor 1 (float)>
// <index of neighbor 2 (unsigned int)> <weight for neighbor 2 (float)>
// .....
//
/////////////////////////////////////////////////////////////////////

#include "reformat_graph.h"

// defining a map that goes from graph vertices to parameters ///////////
map<string, unsigned int> graphToParameters;
/////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// to generate the key for the hash_map
// this is used as a map from a (file_id, frame_id) pair to the
// corresponding index in the paramters.
//
//
string 
generate_key(unsigned int t1, unsigned int t2) {

  ostringstream stm1, stm2;

  stm1 << t1;
  stm2 << t2;

  string tmp = stm1.str();
  tmp.append("_");
  tmp.append(stm2.str());

  return tmp;
}


///////////////////////////////////////////////////////////////////////
// To set the default values of some of the config variables.
// Note that all "variables" on the RHS are defined in
// MP_large_scale.h
//
void 
setDefaults() {
  config.verbosity =  VERBOSITY;
  config.outputGraphIsBinary = true;
}


/////////////////////////////////////////////////////////////////////
// to init the nodes in the graph with data from the original graph.
//
//
//
void 
readGraph(node * graph, 
	  const string graphFileName, 
	  const unsigned int verbosity, 
	  const unsigned int nVertices) {

  ifstream iFile;    
  printf("Reading File %s\n", graphFileName.c_str());
  fflush(stdout);
  iFile.open(graphFileName.c_str(), ios::in|ios::binary);
  if (!iFile.is_open()) GError("unable to open " + graphFileName, 1);
  
  unsigned int ii = 0;
  node *ptr = graph;
  
  while (!iFile.eof()) {

    unsigned int file_id = 0, frame_id = 0, numNeighbors = 0; 

    iFile.read((char*)&file_id, sizeof(unsigned int));
    iFile.read((char*)&frame_id, sizeof(unsigned int));
    iFile.read((char*)&numNeighbors, sizeof(unsigned short));
      
    // if the vertex has no neighbors, then simply got to the next vertex.
    if (numNeighbors == 0) 
      continue;
      
    // if the numNeigbors != 0 and file ends abruptly, get out of the loop.
    if (iFile.eof())
      break;
       
    // populate all neighbor information. 
    ptr->NN = numNeighbors; 
    ptr->idx = new unsigned int[numNeighbors];
    ptr->weight = new float[numNeighbors];
    for (unsigned short i = 0; i < numNeighbors; i++) {

      unsigned int t1, t2;
      float t3;
	
      iFile.read((char*)&t1, sizeof(unsigned int));
      iFile.read((char*)&t2, sizeof(unsigned int));
      iFile.read((char*)&t3, sizeof(float));
      
      ptr->idx[i] =  graphToParameters[generate_key(t1, t2)];
      ptr->weight[i] = t3;
      
    }
    ++ii; ++ptr;

    if ( !(ii % 100000) ) {
      fprintf(stderr,":");
      fflush(stderr);
    }
  }     
  iFile.close();            
  fprintf(stderr,"\n");
}



////////////////////////////////////////////////////////////////////
// to build a map from <file_id, frame_id> => graph_index
//
// this is necessary because we do not store the file and frame id's 
// simply map them to the index within the graph. 
//
// number of vertices in the graph is returned.
//
// GRAPH IS ASSUMED TO BE BINARY.
//
void
buildMap(const string name, 
	 const unsigned int verbosity) {


  fprintf(stderr, "Building Map for Graph File ... %s\n", name.c_str());

  ifstream iFile;    
  iFile.open(name.c_str(), ios::in|ios::binary);

  unsigned int ii = 0;

  // check if the file has been opened properly.
  if (iFile.is_open()) {

    while (!iFile.eof()) {

      unsigned int file_id = 0, frame_id = 0, 
	numNeighbors = 0;

      iFile.read((char*)&file_id, sizeof(unsigned int));
      iFile.read((char*)&frame_id, sizeof(unsigned int));
      iFile.read((char*)&numNeighbors, sizeof(unsigned short));
      
      string key = generate_key(file_id, frame_id);

      // if the vertex has no neighbors, then simply quit. 
      if (numNeighbors == 0) 
	continue;     
      
      if (iFile.eof())
	break;
     

      // now store the entry in the map -- key step, build the map.
      graphToParameters[key] = ii;
      ii++;
      
      // basically skip over the rest of the information for now. 
      unsigned int offset = numNeighbors*(2*sizeof(unsigned int) 
					  + sizeof(float));
      iFile.seekg(offset, ios_base::cur);

      if ( !(ii % 1000000) ) {
	//printf("Done with %d vertices\n", ii);
	fprintf(stderr,":");
	fflush(stderr);
      }
      
    }

    iFile.close();            

  }
  else 
    GError("unable to open " + name, 1);

  fprintf(stderr,"\n");
  fflush(stderr);
  
}


///////////////////////////////////////////////////////////
// get statistics about the graph -- this is really for 
// debugging..
//
//
void 
getStats(node *graph, unsigned int numNodesInGraph) {

  unsigned int numLabeled = 0;
  unsigned int numUnLabeled = 0;
  unsigned int numDevSet = 0;
  unsigned int numNodes = 0;

  for (unsigned int i = 0; i < numNodesInGraph; i++) {
    node *tmp = &graph[i];

    if (tmp->labeled == 'l') 
      numLabeled++;
    else if (tmp->labeled == 'd')
      numDevSet++;
    else 
      numUnLabeled++;
    
    numNodes++;

  }

  cout << "Number of Nodes = " << numNodes << endl;
  cout << "Number of Labeled = " << numLabeled << endl;
  cout << "Number of Unlabeled = " << numUnLabeled << endl;
  cout << "Number of Dev Set = " << numDevSet << endl;

}

/////////////////////////////////////////////////////////////////
// a debug routine that given idx, returns information
// about that node in the graph. 
// Information includes, number of NN's, idx & weights of NN's
//
void 
lookAtNode(const node *graph, unsigned int idx) {
  
  cout << "Node #" << idx << endl;
  cout << "Positions -- " << graph[idx].index  << endl;
  for (unsigned int i = 0; i < graph[idx].NN; i++) {
    cout << "NN_idx = " << graph[idx].idx[i] << " NN_weight = " 
	 << graph[idx].weight[i] << endl;
  }
}


///////////////////////////////////////////////////////////////////////
// to write the graph to an ascii/binary file in the 
// following format.  
// 
// <num_of_vertices>
// vertex_id NN <NN_1_info> <NN_2_info> ... where 
// <NN_n_info> is file_id frame_id weight for the N^{th} nearest 
//
// note that this is going to write the wieghts as it is, without
// any transformation.
//
void writeGraph(node *graph,
		string outputGraphFile, 
		bool outputFileIsBinary,
		unsigned int numNodesInGraph) { 

  node *ptr = graph;
  unsigned int counter = 0;
  ofstream oFile;

  if (outputFileIsBinary) { 
    // write output in binary format
    cout << "writing output in binary format" << endl;
    oFile.open(outputGraphFile.c_str(), ios::out | ios::binary);

    if (!oFile.is_open()) // check if the file is open
      GError("unable to open file " + outputGraphFile + " for writing", 1);

    oFile.write((char*)&(numNodesInGraph), sizeof(unsigned int));

    while (counter < numNodesInGraph) {

      oFile.write((char*)&(counter), sizeof(unsigned int));
      oFile.write((char*)&(ptr->NN), sizeof(unsigned short)); 
      
      // now info. for all NN.
      for (int j = 0; j < ptr->NN; j++) {
	oFile.write((char*)&(ptr->idx[j]), sizeof(unsigned int));
	oFile.write((char*)&(ptr->weight[j]), sizeof(float));
      }
      ++ptr; ++counter;
    }
    oFile.close();    
  }
  else { 
    // ascii output 
    cout << "writing output in ascii format" << endl;
    oFile.open(outputGraphFile.c_str(), ios::out);  
    
    if (!oFile.is_open()) // check if the file is open
      GError("unable to open file " + outputGraphFile + " for writing", 1);
    
    oFile << numNodesInGraph << endl;

    while (counter < numNodesInGraph) {
      oFile << counter << " " << ptr->NN << " ";

      // now the info over all NN's
      for (int j = 0; j < ptr->NN; j++) {
	oFile << ptr->idx[j] << " ";
	oFile << ptr->weight[j] << " ";
      }
      oFile << endl;
      ++ptr; ++counter;
    }
    oFile.close();    
  }
}


////////////////////////////////////////////////////////////////////////
// to free memory associated with the graph
//
//
void
freeMemory(node *graph, 
	   unsigned int numNodesInGraph) {

  node *tmp = graph;

  for (unsigned int i = 0; i < numNodesInGraph; i++, tmp++)  {
    
    delete[] tmp->weight;
    delete[] tmp->idx;
    
  }

}


////////////////////////////////////////////////////////////////////////
// and the most imp. function of them all.
//
//
//
int 
main(int argc, char *argv[]) {


  // set the defaults
  setDefaults();

  // process the command line arguments
  processCmdArgs(argc, argv);
  
  // build the graph //////////////////////////////////////////////////

  // first build the map from <file, frame> => index //
  buildMap(config.inputGraphName, config.verbosity);

  // number of vertices in the graph.
  unsigned int numNodesInGraph = graphToParameters.size();
  printf("Read Graph with %d vertices\n", numNodesInGraph);
  
  // now populate the actual data structure.// 
  node *graph = new node[numNodesInGraph];

  readGraph(graph, 
	    config.inputGraphName, 
	    config.verbosity, 
	    numNodesInGraph);

  // we don't need this anymore. 
  graphToParameters.clear();

  //////////////////////////////////////////////////////////////////////

  // write graph //////////////////////////////////////////////////////
  writeGraph(graph,
	     config.outputGraphName, 
	     config.outputGraphIsBinary,
	     numNodesInGraph);

	
  // free up all the memory that has been uses /////////////////////////////

  freeMemory(graph, 
	     numNodesInGraph);

  delete[] graph;

  ///////////////////////////////////////////////////////////////////////////

  return(0);

}

