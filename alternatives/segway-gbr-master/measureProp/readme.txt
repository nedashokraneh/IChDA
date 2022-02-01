Measure/Label Propagation:
----------------------------

University of Washington, Seattle
Dept. of Electrical Engineering

The following copyright notice applies to all files contained
within the measure propagation source code distribution:
 
Copyright (c) 2010, Amar Subramanya, Jeff Bilmes, and the University
of Washington, Seattle

Permission to use, copy, modify, and distribute this software and its
documentation for any non-commercial purpose and without fee is hereby
granted, provided that the above copyright notice appears in all
copies.  The University of Washington, Seattle make no representations
about the suitability of this software for any purpose. It is provided
"as is" without express or implied warranty.


Any questions, please send mail to:

       Amar Subramanya <asubram@ee.washington.edu>
       Jeff Bilmes <bilmes@ee.washington.edu>

or see the MP web page

        http://ssli.ee.washington.edu/mp/

or look in the code.

-------------------------

This code distribution contains a C++ parallel implementation of
measure propagation, a recently introduced graph-based semi-supervised
learning algorithm. Also, the label propagation algoirthm is 
included as well.

This file contains a (very) brief description of the things you need
to run the code. See the directory ./docs which contains a set of
papers describign the algorithm.

Running MP_large_scale on the command line without any arguments will
give you a list of arguments with a short explanation for each one of
them.  This document is intended to provide a bit more information.

(a) Graph: Currently only binary format is supported. The graph needs
to be in the following form --

<number_of_vertices (unsigned int)>
<vertex id (unsigned int)> <numNeighbors (unsigned short)>
<index of neighbor 1 (unsigned int)> <weight for neighbor 1 (float)>
<index of neighbor 2 (unsigned int)> <weight for neighbor 2 (float)>
.....

All ints are presumed sized to be machine words (so watch out cross
platform issues).

Currently no sanity checks are done while reading the graph except to
ensure that weight >= 0; So if your graph has bugs or
misspecification, e.g., a vertex refers to a neighbor that does not
exist, this can lead to problems or even complete garbage later during
inference.

Henceforth we will use N to denote the number of vertices in the
graph.  Also see point (f).


(b) Transduction File: Once again this is a binary file that contains
N (machine) ints, one number for each vertex of the graph that encodes
the following: 
   if the number = 1, then vertex is labeled; 
   if the number = -2 then vertex belongs to dev set;
   else, the vertex is unlabeled (member of the test set).

- If a vertex is labeled then its label is used during inference.

- If vertex belongs to the dev set, the code computes the error over
those vertices and prints them out if -printAccuracy is set to true.
Thus labels for vertices that belong to the dev set are not used
during inference but may be for example, used for tuning
hyperparameters.

IMPORTANT: THE LENGTH OF THE TRANSDUCTION FILE SHOULD BE IDENTICAL TO
THE NUMBER OF VERTICES IN THE GRAPH (if this is not the case,
unpredictable results could occur).

(c) -labelFileList or -labelFile: There are two ways of specifying the
labels.  -labelFileList can be used to specify an ascii file
containing a list of paths to a set of label files. Each file in the
list should be binary and is read in a serial fashion, windowed if
-nWinSize > 1 and then the labels are assigned to each vertex in the
graph. -labelFile may be used to directly specify the path of a binary
file containing all the labels. The labels are assumed to type int.

The reason for allowing the labels in these two formats is as follows.
For speech recognition, the labels often come in the form of one file
per utternace, which contains a list of vectors x1 x2 ... xN and a
list of labels l1 l2 ... lN. Rather than asking the user to bundle all
of these utterances up into one file, we allow the user to give the
set of utterance files (i.e., the labelFileList option) and the
program will glue things together as appropriate (note that the
nWinSize argument acts as a truncation, as if the vector consists of a
window of speech acoustics, then we don't use the first and last
(nWinSize-1)/2 labels in each labelFileList.

The labels can either be distributions (measure labels) or 'hard' labels
(zero entropy labels).  If the "-measureLabels true" option is given,
MP run in measure labels mode; otherwise, it will run in 
zero entropy mode.

The format of the zero entropy label file is (each value is an int):
<label of node1>
<label of node2>
...

The format of the measure labels file is (each value is a float):
<prob node1=label1> <prob node1=label2> ...
<prob node2=label1> <prob node2=label2> ...
...


(d) -applyRBFKernel (true/false) or (t/f): This can be used to apply the RBF
kernel to the graph weights. Note that -sigma (float) needs be set if this is
set to true.

(e) -weightsFile: This can be used to give a set of values for \mu and
\nu (see NIPS paper in ./docs directory for notation) which are the
trade-off values in the objective. The reason a set of values is given
is that the MP algorithm is run multiple times, once for each
possible combination of mu and nu, and reports the results for each. Each
run is separate and independent, and the facility here is given to run
multiple times for each mu/nu pair is done simply to avoid having
to re-run the program (and re-read a potentially large graph) redundantly
many times over when the only modification is the parameters mu and nu.

The file is expected to be ASCII and should be in the following
format:

<number of mu values> 
<val1> 
<val2>
.....
<number of nu values>
<val1>
<val2>
.....

Having this file makes hyperparameter tuning easier especially if the
size of the graph is large. Without this, repeated calls to
MP_large_scale would be slow as a result of time taken to read the
graph from disk. By specifying a weightsFile, the graph is read only
once and reinitialized for each combination of mu and nu.  

Note that <number of mu values> X <number of nu values> (the Cartesian
product) number of runs are made, once for each possible mu/nu pair.

(f) -outPosteriorFile: This file is written by measureProp, and contains
the posterior probabilities for each node.  The format is:

<num_vertices (unsigned int)> <num_of_classes (unsigned short)>
<vertex_0 (unsigned int)> <prob_0 (float)>.... <prob_k>
<vertex_1> ......


(f) Folder ./reformat_graph:

Often, particularly in the case of graphs generated using speech
corpora, the graph is of the following form:

// <number_of_vertices (unsigned int)>
// <file id (unsigned int)> <frame id (unsigned int)>
// <numNeighbors (unsigned short)>
// <file id of neighbor 1 (unsigned int)> <frame id of neighbor 1 (unsigned int)>
// <weight for neighbor 1 (float)>
// <file id of neighbor 2 (unsigned int)> <frame id of neighbor 2 (unsigned int)>
// <weight for neighbor 2 (float)>
// ..... 

reformat_graph converts graphs in the above format into graphs in the
format requires by MP_large_scale.

(g) Folder ./data: An example set of data files for the labeled
    portion of STP (72 minutes of speech). This is just the labeled
    portion of the graph, not the complete graph which is much bigger.

(h) As always, all additional questions are answered in the code! :-)

