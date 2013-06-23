LOCAL_LIBS=${HOME}/packages
BOOST_INCLUDE=/Users/jeanne/Downloads/boost_1_53_0
BOOST_LIB=/Users/jeanne/Downloads/boost_1_53_0/stage/lib
CXXFLAGS = -g -std=c++0x -O3 -fopenmp -Wall -I.. -I${BOOST_INCLUDE} -m64 -I/usr/include/eigen3
LDFLAGS = -g -std=c++0x -O3 -fopenmp  -L${BOOST_LIB} -lboost_program_options -lboost_serialization -lgfortran
FFLAGS = -Wall -g -O3  
CXX=g++
FC=gfortran

#-----#
#cxx add -I${LOCAL_LIBS}/include
#ldflags add -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib
#export DYLD_LIBRARY_PATH=/Users/jeanne/Downloads/boost_1_53_0/stage/lib
#-----#

CXXFLAGS += -DEIGEN_DONT_PARALLELIZE

local_objs = log_bilinear_model.o lbfgs.o # lbfgs_wrapper.o

all: train_sgd train_clbl train_mlbl train_huff

-include makefile.depend

#-----------------------#
# Local stuff
#-----------------------#
train_huff: train_huff.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_sgd: train_sgd.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_clbl: train_clbl.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

train_mlbl: train_mlbl.o $(local_objs)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

%.o: %.c
	${CC} $(CFLAGS) -c $< -o $@
#
# %.o: %.f
#	$(FC) $(FFLAGS) -c $< -o $@

.PHONY: depend echo
depend:
	$(CXX) -MM $(CXXFLAGS) *.h *.cc > makefile.depend

clean:
	rm -f *.o

clobber: clean
	rm makefile.depend train_sgd
