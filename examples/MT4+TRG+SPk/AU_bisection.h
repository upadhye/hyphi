//    Copyright 2023 Amol Upadhye
//
//    This file is part of hyphi.
//
//    hyphi is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    hyphi is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with hyphi.  If not, see <http://www.gnu.org/licenses/>.

#include <stdio.h>
#include <math.h>

//debug switch
const int AU_BISECTION_DEBUG = 0;

//max number of iterations and maximum size of parameter array
const int AU_BISECTION_MAXITER = 10000;
const int AU_BISECTION_MAXPLEN = 10000;

//absolute value function for doubles
double AU_BISECTION_FABS(double x){ return(x>0 ? x : -x); }

//function evaluation
double AU_BISECTION_FEVAL
( double (*f)(const double *p), int n, double *pEval, double x){
  pEval[n] = x;
  return f(pEval);
}

//error messages
void AU_BISECTION_ERR01(double xlo, double xhi, double flo, double fhi){
  printf("AU_BISECTION: ERROR 01: CANNOT FIND ZERO.  f(%g)=%g and f(%g)=%g.\n",
	 xlo, flo, xhi, fhi);
  fflush(stdout);
  abort();
  return;
}
void AU_BISECTION_ERR02(){
  printf("AU_BISECTION: ERROR 02: TOO MANY ITERATIONS.\n");
  fflush(stdout);
  abort();
  return;
}

double AU_bisection(double (*f)(const double *p), int plen, int n,
		    const double *p0,
		    double xLo, double xHi, double xTol){
  //Find a zero of f(p) by varying the n'th component of p, 
  //between xLo and xHi, while holding the remaning 
  //components of p fixed to their values in p0.  xTol
  //is a fractional tolerance; the function returns a value
  //of x when 
  //     |hi - lo| < (|hi| + |lo|)*xTol.
  //plen is the length of the vectors p and p0.

  //initialize
  int iter = 0;
  double p1[AU_BISECTION_MAXPLEN];
  for(int i=0; i<plen; i++) p1[i]=p0[i]; //don't modify p0
  double lo=xLo, hi=xHi;
  double flo=AU_BISECTION_FEVAL(f,n,p1,lo), fhi=AU_BISECTION_FEVAL(f,n,p1,hi);

  //sanity check; make sure xHi > xLo, 
  //     and |xHi - xLo| > (|xHi| + |xLo|)*xTol,
  //     and there is guaranteed to be a zero in [xLo,xHi]
  if(xLo > xHi) return(AU_bisection(f,plen,n,p0,xHi,xLo,xTol));
  if(xHi-xLo < xTol*(AU_BISECTION_FABS(xHi)+AU_BISECTION_FABS(xLo))) 
    return(0.5*(xHi+xLo));
  while(iter++ < AU_BISECTION_MAXITER/4 && flo*fhi > 0){ //widen iterval
    double Dx = hi - lo;
    lo -= 0.5*Dx;   flo = AU_BISECTION_FEVAL(f,n,p1,lo);  
    hi += 0.5*Dx;   fhi = AU_BISECTION_FEVAL(f,n,p1,hi);

    if(AU_BISECTION_DEBUG){
      printf("#AU_bisection: widening interval: (%g, %g) f: %g, %g\n",
	     lo, hi, flo, fhi);
      fflush(stdout);
    }
    
  }
  if(flo*fhi > 0){ AU_BISECTION_ERR01(lo,hi,flo,fhi); }

  while( iter++ < AU_BISECTION_MAXITER 
	 && (hi-lo) > xTol*( AU_BISECTION_FABS(xLo)+AU_BISECTION_FABS(xHi) ) ){
    double md = 0.5*(hi+lo);
    double fmd = AU_BISECTION_FEVAL(f,n,p1,md);
    if(fmd*flo < 0){ hi=md; fhi=fmd; }
    else{ lo=md; flo=fmd; }

    if(AU_BISECTION_DEBUG){
      printf("#AU_bisection: guess x=%g, f=%g\n", md, fmd);
      fflush(stdout);
    }
    
  }

  if(iter >= AU_BISECTION_MAXITER){ AU_BISECTION_ERR02(); }

  if(AU_BISECTION_DEBUG){
    printf("#AU_bisection: returning x=%g\n", 0.5*(hi+lo));
    fflush(stdout);
  }
  
  return(0.5*(hi+lo));
}
