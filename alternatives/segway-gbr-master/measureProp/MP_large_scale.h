///////////////////////////////////////////////////////////////////////
//
//
//

#ifndef MP_LARGE_SCALE_H
#define MP_LARGE_SCALE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <assert.h>
#include <vector>
#include <math.h>
#include <pthread.h>
#include <ctime>
#include <algorithm>
#include <sys/resource.h>

#include "miscRoutines.h"

using namespace std;

#define USE_FAST_LOG false // should fast log be used for computing log()
#define TABEXP 20 // fast log lookup table size.

// array for the fast log.
float* mytable;

// array of label distributions in the case where
// we use the general form of labels (i.e., the non zero-entropy labels).
// This is a numLabels x numClasses tables, with fastest index being by classes.
float* globalLabelMatrix;

// structure to hold each of the nodes in the graph.
struct node {
  
  // store the parameters. the parameters are stored in one 
  // long array. so if there are numClasses numbers of classes
  // there are a total of 2*numClasses number of parameters.
  float *_pDist;  

  // number of nearest neighbors 
  unsigned short NN; 

  union {
    // the actual label of this node (store it directly here).
    // which this does add to the memory requirements, it is a whole lot
    // easier to deal with when we do sorting of the nodes.
    unsigned short zeroEntropyLabel;
    
    // Alternatively, a pointer to the array of length numClasses
    // (this is presumed) where the label distribution is. This is
    // used in the case where the -measureLabels command line is set
    // to true.
    float * labelDistPtr;
  };

  // Status of this node's labeled condition. This could be 'u' (for
  // unlabeled), or could be 'l' for labeled, or could be 'd' (for
  // being a member of a dev set).
  char labeled; 

  // nearest NN info
  unsigned int *idx; // list of its NN's
  float *weight; // weights to each of NN's

  // index of this node in the original data structure. 
  unsigned int index;

  // the two following function return pointers to each of the 
  // two paramter arrays (i.e., p and q).
  inline float* pDist() { return _pDist; }
  inline float* qDist() { return _pDist + config.numClasses; }

};



/////////////////////////////////////////////////////////////////////////
// this is just a helper so that we can sort the set of neighbors for 
// each nodes -- USED IN reOrderGraph.cc
//
struct NNIdxWeight {

  unsigned int idx;
  float weight;

};

#endif
