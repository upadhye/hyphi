CCFLAGS=-O3 -fopenmp -std=c++11
PATHS=-I/usr/include -L/usr/lib/x86_64-linux-gnu/
LIBS=-lgsl -lgslcblas

hyphi: hyphi.cc AU_cosmological_parameters.h AU_interp.h AU_tabfun.h AU_ncint.h AU_bisection.h AU_halofit.h
	g++ hyphi.cc -o hyphi $(CCFLAGS) $(PATHS) $(LIBS)

clean:
	$(RM) hyphi


