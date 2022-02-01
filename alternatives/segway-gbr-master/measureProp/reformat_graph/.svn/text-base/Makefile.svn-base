FLAGS = -Wall -ansi -pedantic 
OPTIM = -O3 -mfpmath=sse -ffast-math 
PROFILE = -pg
LINK_FLAGS = -lm

reformat_graph: reformat_graph.cc reformat_graph.h
	g++ ${FLAGS} ${OPTIM} -c reformat_graph.cc -o reformat_graph

debug: reformat_graph.cc reformat_graph.h
	g++ -g ${FLAGS} -c reformat_graph.cc -o reformat_graph

clean:
	rm reformat_graph

