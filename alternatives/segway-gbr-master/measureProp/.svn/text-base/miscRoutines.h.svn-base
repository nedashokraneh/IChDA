
#ifndef MISCROUTINES_H
#define MISCROUTINES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <assert.h>
#include <vector>
#include <string>
#include <math.h>
#include <pthread.h>
#include <ctime>
#include <algorithm>
#include <sys/resource.h>
#include <string.h>

using namespace std;

// to get timing numbers.
// report timing for getrusage (this is taken from GMTK).
//
//
void reportTiming(// input
                  const struct rusage& rus,
                  const struct rusage& rue);


//////////// fast log //////////////////////////////////////////////
// function definitions. //////////////////////////////////////////

// function call that builds the lookup table.
void  do_table(float *lookup_table, const int n);

void GError (string message, int status);

// the fast log function call
float 
lut_log(register float val,
	register const float *lookup_table,
	register const int n);


void setDefaults();
void printHelp();
void processCmdArgs(struct alternatingMinimizationConfig& config,int argc,char **argv);



// default setting of some of the variables. //////////////////////////////
const float SMALL = 1E-10; // to ensure that we never take log(0)
const float CONVERGENCE_CRITERIA = 1E-2; // convergence criteria
const bool GRAPH_IS_BINARY = true; // graph is assumed to be in raw binary format.
const unsigned short NUM_CLASSES = 39; // number of classes.
const unsigned short IGNORE_LABEL = 99; // ignore this label.
const unsigned int VERBOSITY = 0; // verbosity of the outputs.
const unsigned short MAX_ITERS = 300; // maximum of iterations.
const unsigned short MIN_ITERS = 20; // minimum number of iterations to run.
const unsigned short WIN_SIZE = 7; // window size for framing
const unsigned short NUM_THREADS = 1; // default number threads to run
const float SIGMA = 1.0;
const unsigned short MAX_MU_NU_LEN = 100; // maximum length of the mu and nu arrays. 
// self-weight -- this is the weight of the edge from the vertex
// back to itself. if you have an rbf kernel then this is always 
// 1.0 assuming you use a distance metric such that d(x,x) = 0;
const float SELF_WEIGHT = 1.0; 



// structure to hold all the config/input parameter values.
struct alternatingMinimizationConfig {

  string inputGraphName; // name of the graph file
  string inputTransductionFile; // name of the transduction file
  string labelFile; // binary file containing the labels 
  string labelFileList; // ascii file containing paths to labels.
  string weightsFile; // file containing the values for mu and nu
  float sigma; // rbf kernel hyperparameters
  float mu; // graph weight mu
  float nu; // entropy trade-off
  float selfWeight; // weight on the link of a vetex to itself.
  unsigned int totalNumVertices; // number of vertices in the graph.
  unsigned short verbosity; // verbosity of output
  unsigned short maxIters; // maximum number of iterations.
  string outPosteriorFile; // where to store PDist
  string outObjFile; // where to store final objective
  unsigned short numClasses; // number of classes
  bool measureLabels; // if false, the labels for each sample is an int. If true, we use
                      //  a full distribution for each label (i.e., as in full-blown measure prop).
  unsigned short nWinSize; // window size to use for the labels.
  unsigned short numThreads; // number of threads to be running 
  bool reOrderGraph; // should we do lexicographic sort on the graph
  bool useSQL; // whether to use squared loss formualation.
  bool useLP; // whether to use label propagation
  bool readWeightsFile; // whether the weights should be read from the weights file
  bool applyRBFKernel; // whether to apply RBF Kernel.
  bool printAccuracy; // whether to print accuracy on every iteration.

  ///////////////////////////////////////////////////////////////////////
  // To set the default values of some of the config variables.
  // Note that all "variables" on the RHS are defined in
  // MP_large_scale.h
  //
  void setDefaults() {
    sigma = SIGMA;
    verbosity =  VERBOSITY;
    maxIters = MAX_ITERS;
    numClasses = NUM_CLASSES;
    measureLabels = false;
    nWinSize =  WIN_SIZE;
    numThreads = NUM_THREADS;
    reOrderGraph = false; 
    outPosteriorFile = "";
    labelFile = "";
    labelFileList = "";
    mu = 0;
    nu = 0;
    readWeightsFile = false; 
    applyRBFKernel = true;
    useSQL = false;
    useLP = false;
    printAccuracy = false;
    selfWeight = SELF_WEIGHT;
  }

};

// extern ref to global version of the above that must be created
// somewhere.
extern struct alternatingMinimizationConfig config;


#endif
