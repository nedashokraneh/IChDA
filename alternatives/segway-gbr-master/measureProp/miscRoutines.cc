
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


//////////////////////////////////////////////////////////////////////
// In case of error, bail.
//
void GError (string message, int status) {
  printf("%s Quitting....\n", message.c_str());
  exit(status);
}



//////////////////////////////////////////////////////////////////////////
// to get timing numbers.
// report timing for getrusage (this is taken from GMTK).
//
//
void reportTiming(// input
                  const struct rusage& rus,
                  const struct rusage& rue) {

  struct timeval utime;
  double utimef;
  struct timeval stime;
  double stimef;

  /* user time */
  utime.tv_sec = rue.ru_utime.tv_sec - rus.ru_utime.tv_sec ;
  if ( rue.ru_utime.tv_usec < rus.ru_utime.tv_usec ) {
    utime.tv_sec--;
    utime.tv_usec = 1000000l - rus.ru_utime.tv_usec +
      rue.ru_utime.tv_usec;
  } else
    utime.tv_usec = rue.ru_utime.tv_usec -
      rus.ru_utime.tv_usec ;
  utimef = (double)utime.tv_sec + (double)utime.tv_usec/1e6;

  /* system time */
  stime.tv_sec = rue.ru_stime.tv_sec - rus.ru_stime.tv_sec ;
  if ( rue.ru_stime.tv_usec < rus.ru_stime.tv_usec ) {
    stime.tv_sec--;
    stime.tv_usec = 1000000l - rus.ru_stime.tv_usec +
      rue.ru_stime.tv_usec;
  } else
    stime.tv_usec = rue.ru_stime.tv_usec -
      rus.ru_stime.tv_usec ;

  stimef = (double)stime.tv_sec + (double)stime.tv_usec/1e6;
  printf("User: %f, System: %f, CPU %f\n", utimef, stimef, utimef+stimef);

}



//////////// fast log ///////////////////////////////////////////////////
//
// contains the routines for fast log() computation. This is basically 
// the icsi log implementation (do a google search for icsi log)
//

float lut_log(register float val,
	      register const float *lookup_table, 
	      register const int n) {
  
  register int *const exp_ptr = ((int*)&val);
  register int x = *exp_ptr; //x is treated as integer
  register const int log_2 = ((x >> 23) & 255) - 127;//exponent
  x &= 0x7FFFFF; //mantissa
  x = x >> (23-n); //quantize mantissa
  val = lookup_table[x]; //lookup precomputed value
  return ((val + log_2)* 0.69314718); //natural logarithm
  
}

// to construct the lookup table. 
void do_table(float *lookup_table, const int n) {

  float numlog;
  int *const exp_ptr = ((int*)&numlog);
  int x = *exp_ptr; //x is the float treated as an integer
  x = 0x3F800000; //set the exponent to 0 so numlog=1.0
  *exp_ptr = x;
  int incr = 1 << (23-n); //amount to increase the mantissa
  int p=pow(2,n);
  for(int i=0;i<p;++i)
    {
      lookup_table[i] = log2(numlog); //save the log value
      x += incr;
      *exp_ptr = x; //update the float value
    }

}



//////////////////////////////////////////////////////////////////////
// to test the fast log
//
//
// void testFastLog() {

  /*  printf("Table size = 2^%d = (%d)\n",TABEXP,tabsize);
  
  float tmp = 0.0000001;
  for (i = 0;i<9;i++) {
    printf("%0.7f log() = %f ",tmp,log(tmp));
    printf("lut_log() = %f\n",lut_log(tmp,mytable,TABEXP));
    tmp *= 10.0;
    }*/
  
//}


/////////////////////////////////////////////////////////////////////
// print helper information
//
void 
printHelp() {
  printf("******     Usage Details      **************************************\n");
  printf("-inputGraphName:path to input graph (required)\n");
  printf("-transductionFile:file containing which vertices are labeled, unlabeled (required)\n");
  printf("-labelFileList:ascii file contain paths to label files\n");
  printf("-labelFile:binary file containing all the labels\n");
  printf("-maxIters:maximum number of iterations to run (300)\n");
  printf("-numClasses:number of classes (required)\n");
  printf("-measureLabels: use a full distribution, length numClasses, as the labels (optional)\n");
  printf("-numThreads:number of threads (1)\n");
  printf("-nWinSize:window size to be applied to the label file (1)\n");
  printf("-reOrderGraph:whether to re-order graph (false)\n");
  printf("-useSQL:whether to run the squared-loss objective (false)\n");
  printf("-useLP:whether to run the vanilla version of lael progagation algorithm (false)\n");
  printf("-applyRBFKernal:should the RBF kernel be applied to the weights read from the file (true)\n");
  printf("-sigma:width of the RBF kernel (float)(required for RBF Kernel)\n");
  printf("-weightsFile:file containing the set of weight (mu, nu) over which to run a grid search\n");
  printf("-outPosteriorFile:output file for converged posteriors\n");
  printf("-outObjFile:output file for information about final objective\n");
  printf("-printAccuracy:whether to print accuracy results every iteration(false)\n");
  printf("-selfWeight:weight on the self loop(1.0)\n");
  printf("-verbosity:verbosity of the outputs\n");
  printf("******     END     *************************************************\n");
}

//////////////////////////////////////////////////////////////////////
// process the command line arguments
//
void 
processCmdArgs(struct alternatingMinimizationConfig& config,
	       int argc,
	       char **argv)
{
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
    else if (strcmp(argv[mark], "-transductionFile") == 0) {
     if (argc > 1) {
       --argc;
        ++mark;
     }
     config.inputTransductionFile = argv[mark];
    }
    else if (strcmp(argv[mark], "-weightsFile") == 0) {
      if (argc > 1) {
	--argc;
        ++mark;
      }
      config.weightsFile = argv[mark];
    }
    else if (strcmp(argv[mark], "-outPosteriorFile") == 0) {
      if (argc > 1) {
	--argc;
        ++mark;
      }
      config.outPosteriorFile = argv[mark];
    }
    else if (strcmp(argv[mark], "-labelFileList") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.labelFileList = argv[mark];
    } 
    else if (strcmp(argv[mark], "-labelFile") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.labelFile = argv[mark];
    }   
    else if (strcmp(argv[mark], "-sigma") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.sigma = atof(argv[mark]);
      // bail if sigma < 0.0
      if (config.sigma < 0.0)
        GError("sigma needs to >= 0.0", 1);
    }
    else if (strcmp(argv[mark], "-selfWeight") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.selfWeight = atof(argv[mark]);
    }
    else if (strcmp(argv[mark], "-outObjFile") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.outObjFile = argv[mark];
    }
    else if (strcmp(argv[mark], "-mu") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.mu = atof(argv[mark]);

      // bail if mu < 0
      if (config.mu < 0.0)
        GError("mu needs to >= 0.0", 1);

    }
    else if (strcmp(argv[mark], "-nu") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.nu = atof(argv[mark]);

      // bail if nu < 0.0
      if (config.nu < 0.0)
        GError("nu needs to >= 0.0", 1);

    }
    else if (strcmp(argv[mark], "-verbosity") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.verbosity = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-maxIters") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.maxIters = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-nWinSize") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.nWinSize = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-numClasses") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.numClasses = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-measureLabels") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.measureLabels = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
	config.measureLabels = false;
      else {
	printf("Cannot understand if you want to use measureLabels or not....");
	printf("Assuming zero entropy labels.\n");
	config.measureLabels = false;
      }
    }
    else if (strcmp(argv[mark], "-numThreads") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      config.numThreads = atoi(argv[mark]);
    }
    else if (strcmp(argv[mark], "-readWeightsFile") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.readWeightsFile = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
	config.readWeightsFile = false;
      else {
	printf("Cannot understand if you want me do read a weights file or not....");
	printf("not reading weights file\n");
      }

    }
    else if (strcmp(argv[mark], "-applyRBFKernel") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.applyRBFKernel = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
	config.applyRBFKernel = false;
      else {
	printf("Cannot understand if you want me to RBF Kernel or not....");
	printf("Applying RBF Kernel.\n");
      }
    }
    else if (strcmp(argv[mark], "-reOrderGraph") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.reOrderGraph = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
        config.reOrderGraph = false;
      else
        cout << "Cannot understand if you want me do a sort of not...." 
	     << endl << "not sorting" << endl;
    }
    else if (strcmp(argv[mark], "-useSQL") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.useSQL = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
        config.useSQL = false;
      else
        printf("Cannot understand, simply using kl-div\n");
    }
    else if (strcmp(argv[mark], "-useLP") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.useLP = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
        config.useLP = false;
      else
        printf("Cannot understand, simply using kl-d\n");
    }
    else if (strcmp(argv[mark], "-printAccuracy") == 0) {
      if (argc > 1) {
        --argc;
        ++mark;
      }
      tmp = argv[mark];
      if (! ( tmp.compare("true") && tmp.compare("t") ) )
        config.printAccuracy = true;
      else if  (! ( tmp.compare("false") && tmp.compare("f") ) )
        config.printAccuracy = false;
      else
        printf("Cannot understand, not printing accuracy\n");
    }
    else {
      cout << "Unrecognized option " << argv[mark] << "!" << endl << endl;
      printHelp();
      exit(1);
    }
    ++mark;
    --argc;
  }
  
  // check if the required arguments have been specified.
  if ( config.inputGraphName.size() == 0 ||
       config.inputTransductionFile.size() == 0  ||
       ( config.labelFile.size() == 0 && config.labelFileList.size() == 0 ) ) {
    printHelp();
    GError( "Required arguments were not specified,", 1);
  }

  if ( config.weightsFile.size() > 0 && config.mu == 0
       && config.nu == 0 && !config.useLP) {
    printf("-------------------------------------------------------------\n");
    printf("As no mu and/or nu were specified on the the command-line, and ");
    printf("weightsFile was specified, switching to reading weightsFile\n");
    printf("-------------------------------------------------------------\n");
    config.readWeightsFile = true;
  }

  // make sure that nWinSize is odd
  if (config.nWinSize % 2 != 1)
    GError("Window size needs to be odd",1);

  if (config.useLP) {
    config.useSQL = true;
    config.mu = 1.0;
    config.nu = 0.0;
    printf("*******************************************\n");
    printf("Note that you have selected to use Label Propogation (LP).\n");
    printf("As a result the following parameters have been automatically\n");
    printf("modified: mu = 1.0, nu = 0.0\n");
    printf("*******************************************\n");
  }

  if (config.sigma > 0) {
    config.applyRBFKernel = true;
  }
}
