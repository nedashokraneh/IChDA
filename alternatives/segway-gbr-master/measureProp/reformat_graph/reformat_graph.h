///////////////////////////////////////////////////////////////////////
//
// contains the hearder information reformat_graph.cc
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <sys/resource.h>

using namespace std;

const bool GRAPH_IS_BINARY = true; // graph is assumed to be in raw binary format.
const unsigned short NUM_CLASSES = 39; // number of classes.
const unsigned short IGNORE_LABEL = 99; // ignore this label.
const unsigned int VERBOSITY = 0; // verbosity of the outputs.

// structure to hold all the config/input parameter values.
static struct alternatingMinimizationConfig {
  string inputGraphName; // name of the graph file.
  string outputGraphName; // name of the output graph file.
  bool outputGraphIsBinary; // whether the output graph is binary.
  unsigned int totalNumVertices; // number of vertices in the graph.
  unsigned short verbosity; // verbosity of output
}config;

// structure to hold each of the nodes in the graph.
struct node {
  
  // number of nearest neighbors 
  unsigned short NN; 

  // the actual label of this node (store it directly here).
  // which this does add to the memory requirements, it is a whole lot
  // easier to deal with when we do sorting of the nodes.
  unsigned short label;

  // is this node labeled
  char labeled; 

  // nearest NN info
  unsigned int *idx; // list of its NN's
  float *weight; // weights to each of NN's

  // index of this node in the original data structure. 
  unsigned int index;
};

/////////////////////////////////////////////////////////////////////////
// this is just a helper so that we can sort the set of neighbors for 
// each nodes -- USED IN reOrderGraph.cc
//
struct NNIdxWeight {

  unsigned int idx;
  float weight;

};


/////////////////////////////////////////////////////////////////
// bail if necessary.
//
void GError (string message, int status) {
  cerr << message << " Quitting... " << endl;
  exit(status);
}


/////////////////////////////////////////////////////////////////////
// print helper information
//
//
//
void 
printHelp() {

  printf("******     Usage Details      **************************************\n");
  printf("-inputGraphName:path to input graph (required)\n");
  printf("-outputGraphName:path to output graph (required)\n");
  printf("-outputGraphIsBinary:whether the output graph should be in ascii or binary format (true)\n");
  printf("******     END     *************************************************\n");

}

//////////////////////////////////////////////////////////////////////
// process the command line arguments
//
//
void 
processCmdArgs(int argc,char **argv) {

  unsigned int mark = 0;
  string tmp;

  --argc;
  ++mark;

  while (argc) {
    if (strcmp(argv[mark], "-inputGraphName") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.inputGraphName = argv[mark];
    }
    else if (strcmp(argv[mark], "-outputGraphName") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.outputGraphName = argv[mark];
    }
    else if (strcmp(argv[mark], "-verbosity") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.verbosity = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-outputGraphIsBinary") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.outputGraphIsBinary = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
	config.outputGraphIsBinary = false;
      else {
	printf("cannot understand if you want me do read a weights file or not....");
	printf("not reading weights file\n");
      }

    }
    else {
      cout << "Unrecognized option " << argv[mark] << "!" << endl << endl;
      printHelp();
      exit(1);
    }
    ++mark;
    --argc;
  }
  
  // cheeck if the required arguments have been specified.
  if ( config.inputGraphName.size() == 0 ) {
    printHelp();
    GError( "Required arguments were not specified,", 1);
  }

}
