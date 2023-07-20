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

  inline double fmax(double x, double y){ return( (x>y)?(x):(y) ); }
  inline double fmin(double x, double y){ return( (x<y)?(x):(y) ); }
  inline double fmax(const double *x, int len){
    return( (len==2)?(fmax(x[0],x[1])):(fmax(fmax(x,len-1),x[len-1])) ); }
  inline double fmin(const double *x, int len){
    return( (len==2)?(fmin(x[0],x[1])):(fmin(fmin(x,len-1),x[len-1])) ); }
  
  double linInterp(const double *xTab, const double *fTab, double xEval){
    double F = fTab[0] + (fTab[1]-fTab[0])/(xTab[1]-xTab[0])*(xEval-xTab[0]);
    return F;
  }

  //cubic polynomial interpolation
  double cubicInterp(const double *xTab, const double *fTab, double xEval){
    //Input the function f(x), in the form of a table of 4 values fTab[]
    //given at the four points xTab[].  Output the interpolated value
    //of the function at xEval, using a cubic polynomial interpolation.
    
    //make sure that we're interpolating, not extrapolating!
    if(xEval<fmin(xTab,4) || xEval>fmax(xTab,4)){
      std::cout << "cubicInterp WARNING: xEval=" << xEval
		<< " out of bounds.  You are" 
		<< std::endl
		<< "                     extrapolating, not interpolating!" 
		<< std::endl;
      abort();
    }
    
    double F = (xEval-xTab[1])*(xEval-xTab[2])*(xEval-xTab[3])
      /(xTab[0]-xTab[1])/(xTab[0]-xTab[2])/(xTab[0]-xTab[3])*fTab[0] 
      + (xEval-xTab[0])*(xEval-xTab[2])*(xEval-xTab[3])
      /(xTab[1]-xTab[0])/(xTab[1]-xTab[2])/(xTab[1]-xTab[3])*fTab[1] 
      + (xEval-xTab[0])*(xEval-xTab[1])*(xEval-xTab[3])
      /(xTab[2]-xTab[0])/(xTab[2]-xTab[1])/(xTab[2]-xTab[3])*fTab[2] 
      + (xEval-xTab[0])*(xEval-xTab[1])*(xEval-xTab[2])
      /(xTab[3]-xTab[0])/(xTab[3]-xTab[1])/(xTab[3]-xTab[2])*fTab[3];
    return(F);
  }

  int findN(double x, const double *xTable, int tableSize){ 
    //do something simple for now; can use a spiffier algorithm if 
    //large tables are necessary
    int n = 0;
    while(xTable[n+1]<x && n<tableSize-1){n++;}
    return n;
  }

int findN(int nguess, double x, const double *xTable, int tableSize){
  //do something simple for now; can use a spiffier algorithm if
  //large tables are necessary
  int n = (nguess>2 ? nguess-2 : 0);
  if(n > tableSize-1) n = tableSize-1;
  while(xTable[n+1]<x && n<tableSize-1){n++;}
  return n;
}

