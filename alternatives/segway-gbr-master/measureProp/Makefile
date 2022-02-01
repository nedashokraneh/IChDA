# $Id: Makefile,v 1.4 2010/02/18 00:59:48 bilmes Exp $

FLAGS = -Wall -ansi -pedantic -m64 -g
OPTIM = -mfpmath=sse -ffast-math 
PROFILE = -pg
THREAD_FLAGS = -D_REENTRANT
LINK_FLAGS = -lm -lpthread
TAR = /bin/tar

all: MP_large_scale

miscRoutines: miscRoutines.cc
	g++ ${FLAGS} ${OPTIM} -fno-strict-aliasing -c miscRoutines.cc

MP_large_scale: MP_large_scale.cc MP_large_scale.h update_P_and_Q.cc reOrderGraph.cc miscRoutines
	g++ ${FLAGS} ${THREAD_FLAGS} ${OPTIM} -c MP_large_scale.cc 
	g++ -o MP_large_scale ${THREAD_FLAGS} miscRoutines.o MP_large_scale.o ${LINK_FLAGS}

debug: MP_large_scale.cc MP_large_scale.h update_P_and_Q.cc reOrderGraph.cc miscRoutines
	g++ -g ${FLAGS} ${THREAD_FLAGS} -c MP_large_scale.cc 
	g++ -o MP_large_scale ${THREAD_FLAGS} miscRoutines.o MP_large_scale.o ${LINK_FLAGS}

clean:
	rm *.o
	rm MP_large_scale



# use this to make dated source for others.
dist:  EXCLUDE
	$(TAR) cvzXf EXCLUDE - . > ../mp-`date +%a_%b_%d_%Y_%k:%M | sed -e 's, ,,g'`.tar.gz

# files/dirs that should not be contained in src distribution. 
EXCLUDE = \
	EXCLUDE

EXCLUDE: force
	(find $(EXCLUDE) -type d -print -prune ; \
	find . \( -name "*~" -o -name "*~[0-9]*" -o -name "core*" -o -name "*.o" -o -name "*.a" -o -name "#*" -o -name ".#*" -o -name "*_bak" \) -print; \
	find . -name CVS -print; \
	find . -name RCS -print; \
	find . -type d -name old -print; \
	find . \( -name TODO -o -name notes -o -name depends.make \) -print ) | \
	sed 's,^\./,,' > $@


force:

