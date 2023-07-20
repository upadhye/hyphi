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

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spline.h>

#include "AU_ncint.h" //needed for halofit
#include "AU_bisection.h" //needed for halofit
#include "AU_interp.h"
#include "AU_cosmological_parameters.h"
#include "AU_ftb_bahamas.h"

//cosmological parameters object
string parameters_input = "params_hyphi.dat";
cosmological_parameters C(parameters_input);

using namespace std;

//debug flags
const int DEBUG_ALL = 0;
const int DEBUG_LINEAR = 0;
const int DEBUG_TIMERG_EVOL = 0;
const int DEBUG_TIMERG_INTEGRAND = 0;
const int DEBUG_TIMERG_INTEGRAND2 = 0;
const int DEBUG_INTEGRATION = 0;
const int DEBUG_CONVERGENCE = 0;

//output options
const int PREC=12, WIDTH=PREC+8;
const int PRINTA=0, PRINTI=0, PRINTQ=0, PRINTBIAS=0;
const int PRINTPMAT=0, PRINTHEATMAP=0;
const int PRINTCLPP=1;

//external variable to store input A_s for Euclid Emulator 2
double A_s_extern = -1;

//Hubble const. in units of h/Mpc
const double H0h = 0.00033356754857714242474; //H0 / (h/Mpc)

//lensing potential power spectrum integration
const double ellkapmax = 25000;
const double z_star = 1089.80; //decoupling redshift from 1807.06209

const int nu_int = -2;
const double nu = (double)nu_int;

//useful functions: square, cube, quadrature add
double sq(double x){ return x*x; }
double cu(double x){ return x*x*x; }
inline double qadd(double x, double y){ return sqrt(sq(x)+sq(y)); }

////////////////////////////////////////////////////////////////////////////////
//k grid and windowing: nk is number of points in "real" k grid,
//I extend this by a factor of four for fast-pt.  The extended grid:
//   -3nk/2  <= i < -17nk/16 : P=0
//  -17nk/16 <= i < -nk      : P tapered smoothly from 0 to P[0]
//     -nk   <= i < 0        : Power spectrum extrapolated to left
//       0   <= i < nk       : Power spectrum region of interest
//      nk   <= i < 2nk      : Power spectrum extrapolated to right
//     2nk   <= i < 33nk/16  : P tapered smoothly from P[nk-1] to 0
//  33nk/16  <= i < 5nk/2    : P=0

const int nk = 128, np = 4*nk, nshift = (np-nk)/2;
const double kmin=1e-3, kmax=15, lnkmin=log(kmin), lnkmax=log(kmax);
const double dlnk=(lnkmax-lnkmin)/(nk-1), dlnell=log(ellkapmax)/(nk-1);
double lnkArr[nk], kArr[nk], lnellArr[nk], ellArr[nk];

//split up interval between zero-padding and tapering:
//these are measured in units of nk/16, and must add to (np/nk-1)*16
//const int s_padL=7+16, s_tapL=1+8, s_extL=16+8, s_extR=16+8, s_tapR=1+8, s_padR=7+16; //use for np = 8*nk
const int s_padL=7, s_tapL=1, s_extL=16, s_extR=16, s_tapR=1, s_padR=7; //use for np = 4*nk
//const int s_padL=3, s_tapL=1, s_extL=4, s_extR=4, s_tapR=1, s_padR=3; //use for np = 2*nk

const double lnk_pad_min   = lnkmin        - dlnk*nshift;
const double lnk_pad_winLo = lnk_pad_min   + dlnk*nk*s_padL/16;
const double lnk_pad_winLi = lnk_pad_winLo + dlnk*nk*s_tapL/16;
const double lnk_pad_winRi = lnk_pad_winLi + dlnk*(nk*(16+s_extL+s_extR)/16 -1);
const double lnk_pad_winRo = lnk_pad_winRi + dlnk*nk*s_tapR/16;
const double lnk_pad_max   = lnk_pad_winRo + dlnk*nk*s_padR/16;

//include halofit here, as it needs nk
#include "AU_halofit.h"

//power spectrum and Fourier coefficient window functions
inline double W_edge(double x){ return x - sin(2.0*M_PI*x)/(2.0*M_PI); }

double WP(double lnk){
  if(lnk <= lnk_pad_winLo) return 0;
  else if(lnk < lnk_pad_winLi) 
    return W_edge((lnk-lnk_pad_winLo)/(lnk_pad_winLi-lnk_pad_winLo));
  else if(lnk < lnk_pad_winRi) return 1;
  else if(lnk < lnk_pad_winRo) 
    return W_edge((lnk_pad_winRo-lnk)/(lnk_pad_winRo-lnk_pad_winRi));
  return 0;
}

const int nl = 1*np/8, nc = np/2, nr = 7*np/8, Dn = 3*np/8;
double WC(int n){ //coeffs in GSL halfcomplex notation
  if(n<=nl || n>=nr) return 1;
  else if(n<nc) return W_edge(double(nc-n)/Dn);
  else if(n<nr) return W_edge(double(n-nc)/Dn);
  return 1;
}

//integration tolerances
const double eabs_P = 1e-15, erel_P = 1e-06; //eta integration for P(k)

//unique components of A_{acd,bef}
//first 8: 001,bef.  next 6: 111,bef for bef={000, 001, 011, 100, 101, 111}
//number of unique components (P, I, Q) 
const int nUP=3, nUI=14, nELL=3, nUQ=nELL*8, nU=nUP+nUI+nUQ;
const int aU[nUI] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int cU[nUI] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int dU[nUI] = {1,1,1,1,1,1,1,1, 1,1,1,1,1,1};
const int bU[nUI] = {0,0,0,0,1,1,1,1, 0,0,0,1,1,1};
const int eU[nUI] = {0,0,1,1,0,0,1,1, 0,0,1,0,0,1};
const int fU[nUI] = {0,1,0,1,0,1,0,1, 0,1,1,0,1,1};
const int JU[nUI] = {8,9,10,11,12,13,14,15,56,57,59,60,61,63};

//P_{T,jm}(k_i) is a 2-D array PT[n][i], with n determining j and m as follows:
const int j_n[9] = {2,2,2,4,4,4,6,6,8};
const int m_n[9] = {2,1,0,2,1,0,1,0,0};

inline void discard_comments(std::ifstream *file){
  while(file->peek()=='#' || file->peek()=='\n'){file->ignore(10000,'\n');} }

inline int nAI(int a, int c, int d, int b, int e, int f){
  return 32*a + 16*c + 8*d + 4*b + 2*e + f; }

int dummy_jacobian(double t, const double y[], double *dfdy,
                   double dfdt[], void *params){ return GSL_SUCCESS; }

///////////////////////////////////////////////////////////////////////////////
//mode-coupling integrals and vertices
double gamma(int a, int b, int c, double k, double q, double p){

  if(DEBUG_ALL || DEBUG_TIMERG_INTEGRAND2)
    cout << "#gamma begin. Called with a=" << a << ", b=" << b
	 << ", c=" << c << ", k=" << k << ", q=" << q
	 << ", p=" << p << endl;

  double gam = 0, eps_gam = 1e-6;
  if(a==0){
    if(b==0 && c==1) gam = (fabs(p/k)>eps_gam 
			    ? 0.25 * (k*k + p*p - q*q) / (p*p) 
			    : 0);
    if(b==1 && c==0) gam = (fabs(q/k)>eps_gam 
			    ? 0.25 * (k*k + q*q - p*p) / (q*q) 
			    : 0);
  }
  if(a==1 && b==1 && c==1) {
    double k2=k*k, p2=p*p, q2=q*q;
    if(fabs(p/k) > eps_gam && fabs(q/k) > eps_gam)
      gam = 0.25 * k2 * (k2-q2-p2) / (p2*q2);
  }

  if(DEBUG_ALL || DEBUG_TIMERG_INTEGRAND2)
    cout << "#gamma begin. Returning gam=" << gam << endl;

  return gam;
}

///////////////////////////////////////////////////////////////////////////////
//interpolate/extrapolate power spectrum with indices

//high-k extrapolation: use Eisenstein-Hu no-wiggle power spectrum
double T_EH(double k){
  static double Omh2 = C.Omega_m()*sq(C.h()), Obh2 = C.Omega_b()*sq(C.h()),
    Onh2 = C.Omega_nu()*sq(C.h()), Tcmb = C.T_cmb_K();
  static double alpha_G = 1.0  - 0.328*log(431.0*Omh2) * Obh2/Omh2 
    + 0.38*log(22.3*Omh2) * sq(Obh2/Omh2);
  static double r_d = 55.234*C.h() / 
    ( pow(Omh2-Onh2,0.2538) * pow(Obh2,0.1278) * pow(1.0+Onh2,0.3794) );

  double Gamma_eff = C.Omega_m()*C.h() * 
    ( alpha_G + (1.0-alpha_G) / (1.0 + pow(0.43*k*r_d,4)) );
  double q_EH = k * sq(Tcmb/2.7) / Gamma_eff;
  double L_EH = log(2.0*M_E + 1.8*q_EH);
  double C_EH = 14.2 + 731.0/(1.0+62.5*q_EH);
  return L_EH / (L_EH + sq(q_EH)*C_EH);
}

//power spectrum
double Pab(int a, int b, double k, const double *lnPk){
  
  if(DEBUG_ALL || DEBUG_TIMERG_INTEGRAND2)
    cout << "#Pab begin. Called with a=" << a << ", b=" << b << ", k=" << k
	 << ", lnPk[0]=" << lnPk[0] << endl;

  //sanity check
  if(a<0 || a>1 || b<0 || b>1 || k<=0){
    cout << "ERROR: Incorrect inputs to Pab." << endl;
    cout << "       a=" << a << ", b=" << b << ", k=" << k << endl;
    abort();
  }

  //interpolate to find power spectrum value
  double lnP = 1e100, lnk = log(k);
  int nguess = (lnk-lnkArr[0])/dlnk; 
  int n=findN(nguess, lnk, lnkArr, nk), 
    interp_type = 0; //0=cubic; +/-1=linear; 2=extrap
  if(n==0) interp_type = -1;
  if(n==nk-2) interp_type = 1;
  if(n>=nk-1 || lnk>lnkArr[nk-1]) interp_type = 2;

  switch (interp_type) {
  case(0):
    lnP = cubicInterp(&lnkArr[n-1], &lnPk[(a+b)*nk + n-1], lnk);
    break;
  case(1):
    lnP = linInterp(&lnkArr[n], &lnPk[(a+b)*nk + n], lnk);
    break;
  case(2):
    n = nk-1;
    lnP = lnPk[(a+b)*nk + n] + (C.n_s()-3.0)*(lnk - lnkArr[n]);
    //lnP = lnPk[(a+b)*nk + n] + C.n_s()*(lnk - lnkArr[n]) 
      //+ 2.0 * log( T_EH(k) / T_EH(kArr[n]) ); 
    break;  
  case(-1): //also extrapolates to the left
    lnP = linInterp(&lnkArr[n], &lnPk[(a+b)*nk + n], lnk);
    break;
  default:
    cout << "ERROR: Invalid interpolation case in Pab." << endl;
    abort();
    break;
  }

  if(DEBUG_ALL || DEBUG_TIMERG_INTEGRAND2)
    cout << "#Pab end. Returning P=" << exp(lnP) << endl;

  return exp(lnP);
}

//interpolated power spectrum from Mira-Titan IV emulator
double Pm_MT4(double eta, double k){

  static int init = 0;
  const double zmax_MT4 = 2.02;
  const int nk_MT4 = 351, nz_MT4 = 25;
  static tabulated_function lnP_lnalnk;

  if(!init){
    double h0=C.h(), s8=C.sigma_8(), ns=C.n_s(), w0=C.w0(), wa=C.wa(), 
      om=C.Omega_m()*h0*h0, ob=C.Omega_b()*h0*h0, on=C.Omega_nu()*h0*h0;
    string mt4_command = "./AU_MT4_wrapper.bash";
    mt4_command += " " + to_string(om);
    mt4_command += " " + to_string(ob);
    mt4_command += " " + to_string(s8);
    mt4_command += " " + to_string(h0);
    mt4_command += " " + to_string(ns);
    mt4_command += " " + to_string(w0);
    mt4_command += " " + to_string(wa);
    mt4_command += " " + to_string(on);
    int mt4_status = system(mt4_command.c_str());
    lnP_lnalnk.initialize("MT4_emu_output.dat", nz_MT4, nk_MT4, 3, 0, 1, 2);
    init = 1;
  }

  double lna=eta+log(C.a_in()), ae=exp(lna), ze=1.0/ae-1.0;
  if(ze>zmax_MT4) return 0; //out of bounds
  return exp(lnP_lnalnk(lna,log(k))) * ae*ae;
}

//interpolated power spectrum from Euclid Emulator 2
//  (see 2010.11288 and github.com/miknab/EuclidEmulator2)
double Pm_EE2(double eta, double k, double A_s_input){

  static int init = 0;
  const int nk_EE2 = 613, nz_EE2 = 40;
  const double zmin_EE2 = 0, zmax_EE2 = 10;
  static tabulated_function lnPPl_lnalnk;

  if(!init){
    string ee2_command = "./AU_EE2_wrapper.bash";
    ee2_command += " " + to_string(C.Omega_m());
    ee2_command += " " + to_string(C.Omega_b());
    ee2_command += " " + to_string(A_s_input*1e9) + "e-9";
    ee2_command += " " + to_string(C.h());
    ee2_command += " " + to_string(C.n_s());
    ee2_command += " " + to_string(C.w0());
    ee2_command += " " + to_string(C.wa());
    ee2_command += " " + to_string(C.Omega_nu()*C.h()*C.h() * 93.25);
    int ee2_status = system(ee2_command.c_str());
    lnPPl_lnalnk.initialize("EE2_emu_output.dat", nz_EE2, nk_EE2, 3, 0, 1, 2);
    init = 1;
  }

  double lna=eta+log(C.a_in()), ae=exp(lna), ze=1.0/ae-1.0, Plin=C.Plin(ze,k);
  if(ze<zmin_EE2 || ze>zmax_EE2) return 0; //out of bounds
  return exp(lnPPl_lnalnk(lna,log(k))) * Plin;
}

///////////////////////////////////////////////////////////////////////////////
//translate from 14 unique components of I/A to full 64-component arrays
int I64(const double *Ij, double *Iacdbef){
  for(int i=0; i<64*nk; i++) Iacdbef[i]=0;

  for(int i=0; i<nk; i++){

    for(int j=0; j<nUI; j++){ Iacdbef[JU[j]*nk+i]=Ij[j*nk+i]; }

    Iacdbef[16*nk+i] = Iacdbef[8*nk+i];
    Iacdbef[18*nk+i] = Iacdbef[9*nk+i];
    Iacdbef[17*nk+i] = Iacdbef[10*nk+i];
    Iacdbef[19*nk+i] = Iacdbef[11*nk+i];
    Iacdbef[20*nk+i] = Iacdbef[12*nk+i];
    Iacdbef[22*nk+i] = Iacdbef[13*nk+i];
    Iacdbef[21*nk+i] = Iacdbef[14*nk+i];
    Iacdbef[23*nk+i] = Iacdbef[15*nk+i];
    Iacdbef[58*nk+i] = Iacdbef[57*nk+i];
    Iacdbef[62*nk+i] = Iacdbef[61*nk+i];
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
//Legendre polynomials
double Pleg(int ell, double x){
  switch(ell){
  case 0:
    return 1.0;
    break;
  case 1:
    return x;
    break;
  case 2:
    return 1.5*x*x - 0.5;
    break;
  case 3:
    return 2.5*x*x*x - 1.5*x;
    break;
  case 4:
    return 4.375*x*x*x*x - 3.75*x*x + 0.375;
    break;
  default:
    cout << "ERROR in Pleg: enter more polynomials!"<< endl;
    abort();
    break;
  }
  return 1e300; //shouldn't ever get here.
}

///////////////////////////////////////////////////////////////////////////////
//translate between Q^{(\ell)}_{abc}(k) and P_{bis,j}(k) for j=2,4,6.
//  ellm1 = ell - 1.  See NOTES 2017-03-31.

inline double QQ(int ellm1, int a, int b, int c, int i_k, const double *y){
  return y[(nUP + nUI + (ellm1)*8 + 4*a + 2*b + c)*nk + i_k];}

double Pbisj(int i, int j_mu, int m_b, const double *y){
  double Qcomb = 0;//linear comb. of Q values

  switch(j_mu){
  case 2:
    Qcomb = (m_b==2) * (
			-2.0*QQ(0, 0,1,0, i,y) + (4.0/3.0)*QQ(1, 0,1,0, i,y)
			)
      + (m_b==1) * (
		    (4.0/3.0)*QQ(1, 0,1,1, i,y) + (6.0/5.0)*QQ(2, 0,1,1, i,y)
		    );
    break;
  case 4:
    Qcomb = (m_b==1) * (
			-2.0*QQ(0, 1,1,0, i,y) + (4.0/3.0)*QQ(1, 1,1,0, i,y) 
			- 2.0*QQ(0, 0,1,1, i,y) - 2.0*QQ(2, 0,1,1, i,y)
			)
      + (m_b==0) * (
		    (4.0/3.0)*QQ(1, 1,1,1, i,y) + (6.0/5.0)*QQ(2, 1,1,1, i,y)
		    );
    break;
  case 6:
    Qcomb = (m_b==0) * (-2.0*QQ(0, 1,1,1, i,y) - 2.0*QQ(2, 1,1,1, i,y));
    break;
  default:
    cout << "ERROR in Pbisj: j_mu=" << j_mu << " invalid; choose 2,4,6"<< endl;
    abort();
    break;
  }

  return M_PI * kArr[i] * Qcomb;
}

////////////////////////////////////////////////////////////////////////////////
// fast-pt computations of A, R, P_T 
// (see McEwen, Fang, Hirata, Blazek 1603.04826)

////////// functions derived from Gamma

int g_MFHB(double mu, double reKappa, double imKappa, double *g_out){

  //compute Gamma function
  gsl_sf_result lnGtop, lnGbot, argTop, argBot;
  gsl_sf_lngamma_complex_e(0.5*(mu+reKappa+1), 0.5*imKappa,  &lnGtop, &argTop);
  gsl_sf_lngamma_complex_e(0.5*(mu-reKappa+1), -0.5*imKappa, &lnGbot, &argBot);
  
  //g_out[] = { |g|, arg(g) }
  g_out[0] = exp(lnGtop.val - lnGbot.val);
  g_out[1] = argTop.val - argBot.val;
  return 0;
}

int f_MFHB(double reRho, double imRho, double *f_out){

  double g[2], pre = 0.5*sqrt(M_PI) * pow(2.0,reRho);
  g_MFHB(0.5,reRho-0.5,imRho,g);
  f_out[0] = pre * g[0];
  f_out[1] = imRho*M_LN2 + g[1];
  return 0;
}

//frontends for the above functions
int f_MFHB(int alpha, int beta, int h, double *f){
  int n = (h<=np ? h : h-2*np);
  //typo in paper?  Real part in FASTPT code is p+1=-4-2*nu-alpha-beta not p
  return f_MFHB(-4.0-2.0*nu-(double)(alpha+beta), -2.0*M_PI*n/(dlnk*np), f);
}

int g_reg_MFHB(int m, double *g_out){ //regularized version for ell=0, alpha=-2
  int n = (m<=np/2 ? m : m-np);
  return f_MFHB(nu, 2.0*M_PI*n/(dlnk*np), g_out); 
}

int g_MFHB(int ell, int alpha, int m, double *g){
  if(m==0 && alpha==ell-nu_int){ g[0]=0; g[1]=0; return 0; }
  if(alpha==-2 && ell==0) return g_reg_MFHB(m,g);
  int n = (m<=np/2 ? m : m-np);
  return g_MFHB(0.5+(double)ell, 1.5+nu+(double)alpha, 2.0*M_PI*n/(dlnk*np), g);
}

////////// frontends for gsl fft routines

//forward fft, output replaces input array
int fft(double *x, int N){ return gsl_fft_real_radix2_transform(x,1,N); }

//inverse fft
int ifft(double *x, int N){ return gsl_fft_halfcomplex_radix2_inverse(x,1,N); }

//backward fft, identical to ifft except for lack of normalization factor
int bfft(double *x, int N){ return gsl_fft_halfcomplex_radix2_backward(x,1,N); }

//convolution fo real functions; assume arrays of equal length
int convolve(int N, double *in0, double *in1, double *out){
  fft(in0,N);
  fft(in1,N);

  //out is now in halfcomplex format
  out[0] = in0[0]*in1[0];
  out[N/2] = in0[N/2]*in1[N/2];

  for(int i=1; i<N/2; i++){
    out[i] = in0[i]*in1[i] - in0[N-i]*in1[N-i];
    out[N-i] = in0[i]*in1[N-i] + in0[N-i]*in1[i];
  }

  ifft(out,N);
  return 0;
}

//convolution for halfcomplex functions; assume arrays already of equal length 
int iconvolve(int N, double *in0, double *in1, double *out){
  ifft(in0,N);
  ifft(in1,N);
  for(int i=0; i<N; i++) out[i] = in0[i] * in1[i] * N;
  fft(out,N);
  return 0;
}

//convolution for complex arrays, with even elements respresenting
//real values and odd elements imaginary values
//For linear convolution, final half of input arrays should be zero-padded.
int cconvolve(int N, double *in0, double *in1, double *out){
  gsl_fft_complex_radix2_forward(in0,1,N);
  gsl_fft_complex_radix2_forward(in1,1,N);
  for(int i=0; i<N; i++){
    out[2*i] = in0[2*i]*in1[2*i] - in0[2*i+1]*in1[2*i+1];
    out[2*i+1] = in0[2*i+1]*in1[2*i] + in0[2*i]*in1[2*i+1];
  }
  gsl_fft_complex_radix2_inverse(out,1,N);
  return 0;
}

int convolve_bruteforce(int N, double *in0, double *in1, double *out){
  for(int i=0; i<N; i++) out[i] = 0;

  for(int n=0; n<N; n++){
    for(int m=0; m<=n; m++) out[n] += in0[m] * in1[n-m];
    for(int m=n+1; m<N; m++) out[n] += in0[m] * in1[N+n-m];
  }

  return 0;
}

//regularized J_{2,-2,0} from McEwen++ 1603.04826
int Jreg_MFHB(const double *Palpha, const double *Pbeta, double *Ji){
  const int alpha=2, beta=-2, ell=0;

  //fft, after multiplying by power law 
  double ca[np], cb[np];
  for(int i=0; i<np; i++){
    double lnk = lnk_pad_min + dlnk*i, k_nnu = exp(-nu*lnk);
    ca[i] = Palpha[i] * k_nnu;
    cb[i] = Pbeta[i] * k_nnu;
  }
  fft(ca,np);
  fft(cb,np);
  for(int i=0; i<np; i++){
    double win = WC(i);
    ca[i] *= win;
    cb[i] *= win;
  }

  double cga[4*np], cgb[4*np], ga[2], gb[2];
  for(int i=0; i<4*np; i++){ cga[i]=0; cgb[i]=0; }

  g_MFHB(ell,alpha,0,ga);
  g_MFHB(ell,beta,0,gb);
  ga[0] *= pow(2.0,1.5+nu+alpha);
  ga[1] += 2.0*M_PI     * 0     /(dlnk*np) * M_LN2;
  cga[0] = ca[0]*ga[0]*cos(ga[1]);
  cga[1] = ca[0]*ga[0]*sin(ga[1]);
  cgb[0] = cb[0]*gb[0]*cos(gb[1]);
  cgb[1] = cb[0]*gb[0]*sin(gb[1]);

  //halfcomplex convolution
  for(int i=1; i<np/2; i++){
    g_MFHB(ell,alpha,i,ga);
    g_MFHB(ell,beta,i,gb);
    ga[0] *= pow(2.0,1.5+nu+alpha);
    ga[1] += 2.0*M_PI*i/(dlnk*np)* M_LN2;

    //fullcomplex convolution for halfcomplex ca and cb
    cga[2*i] = ca[i]*ga[0]*cos(ga[1]) - ca[np-i]*ga[0]*sin(ga[1]);
    cga[2*i+1] = ca[i]*ga[0]*sin(ga[1]) + ca[np-i]*ga[0]*cos(ga[1]);
    cga[4*np-2*i] = ca[i]*ga[0]*cos(ga[1]) - ca[np-i]*ga[0]*sin(ga[1]);
    cga[4*np-2*i+1] = -ca[i]*ga[0]*sin(ga[1]) - ca[np-i]*ga[0]*cos(ga[1]);
    cgb[2*i] = cb[i]*gb[0]*cos(gb[1]) - cb[np-i]*gb[0]*sin(gb[1]);
    cgb[2*i+1] = cb[i]*gb[0]*sin(gb[1]) + cb[np-i]*gb[0]*cos(gb[1]);
    cgb[4*np-2*i] = cb[i]*gb[0]*cos(gb[1]) - cb[np-i]*gb[0]*sin(gb[1]);
    cgb[4*np-2*i+1] = -cb[i]*gb[0]*sin(gb[1]) - cb[np-i]*gb[0]*cos(gb[1]);
  }

  //convolve to get C_h, then IFFT
  double C_h_cmplx[4*np], C_h[2*np], Cf_h[2*np], f[2];
  cconvolve(2*np,cga,cgb,C_h_cmplx);
  
  //recover halfcomplex C_h
  C_h[0] = C_h_cmplx[0];
  C_h[np] = C_h_cmplx[2*np];
  for(int i=0; i<np; i++){
    C_h[i] = C_h_cmplx[2*i];
    C_h[2*np-i] = C_h_cmplx[2*i+1];
  }

  f_MFHB(alpha,beta,0,f);
  Cf_h[0] = C_h[0] * f[0] * cos(f[1]);

  for(int i=1; i<=np; i++){
    double MC = qadd(C_h[i],C_h[2*np-i]), AC = atan2(C_h[2*np-i],C_h[i]);
    if(i==np){ MC = C_h[np]; AC = 0; }
    f_MFHB(alpha,beta,i,f);

    double MCf = MC * f[0], ACf = AC + f[1];
    if(i==np) ACf = 0;
    Cf_h[2*np-i] = MCf * sin(ACf);
    Cf_h[i] = MCf * cos(ACf);
  }

  bfft(Cf_h,2*np); //now it's real

  //assemble J array
  double sl = (ell%2==0 ? 1 : -1);
  double pre = sl / (2.0*M_PI*M_PI*np*np) * sqrt(2.0/M_PI);
  for(int i=0; i<np; i++){
    double ki=exp(lnk_pad_min+dlnk*i), k_npm2=pow(ki,3.0+2.0*nu+alpha+beta);
    Ji[i] = pre * k_npm2 * Cf_h[2*i];
  }

  return 0;
}

//J_{alpha,beta,ell} from McEwen++ 1603.04826
int J_MFHB(int alpha, int beta, int ell, 
           const double *Palpha, const double *Pbeta, double *Ji){

  //do we want the regularized version?
  if(ell==0 && alpha==2 && beta==-2) return Jreg_MFHB(Palpha,Pbeta,Ji);
  if(ell==0 && alpha==-2 && beta==2) return Jreg_MFHB(Pbeta,Palpha,Ji);

  //fft, after multiplying by power law
  double ca[np], cb[np];
  for(int i=0; i<np; i++){
    double lnk = lnk_pad_min + dlnk*i, k_nnu = exp(-nu*lnk);
    ca[i] = Palpha[i] * k_nnu;
    cb[i] = Pbeta[i] * k_nnu;
  }
  fft(ca,np);
  fft(cb,np);
  for(int i=0; i<np; i++){ 
    double win = WC(i);
    ca[i] *= win;
    cb[i] *= win;
  }

  //combine c with g, into 2N element array
  double cga[2*np], cgb[2*np], ga[2], gb[2];
  for(int i=0; i<2*np; i++){ cga[i]=0; cgb[i]=0; }

  g_MFHB(ell,alpha,0,ga);
  g_MFHB(ell,beta,0,gb);
  cga[0] = ca[0]*ga[0];
  cgb[0] = cb[0]*gb[0];

  for(int i=1; i<np/2; i++){
    g_MFHB(ell,alpha,i,ga);
    g_MFHB(ell,beta,i,gb);

    //use for fft convolution
    cga[i] = ca[i]*ga[0]*cos(ga[1]) - ca[np-i]*ga[0]*sin(ga[1]);
    cga[2*np-i] = ca[i]*ga[0]*sin(ga[1]) + ca[np-i]*ga[0]*cos(ga[1]);
    cgb[i] = cb[i]*gb[0]*cos(gb[1]) - cb[np-i]*gb[0]*sin(gb[1]);
    cgb[2*np-i] = cb[i]*gb[0]*sin(gb[1]) + cb[np-i]*gb[0]*cos(gb[1]);
  }

  //convolve to get C_h, then IFFT
  double C_h[2*np], Cftau_h[2*np], f[2];
  iconvolve(2*np,cga,cgb,C_h);

  f_MFHB(alpha,beta,0,f);
  Cftau_h[0] = C_h[0] * f[0] * cos(f[1]);
  
  for(int i=1; i<=np; i++){
    double MC = qadd(C_h[i],C_h[2*np-i]), AC = atan2(C_h[2*np-i],C_h[i]);
    if(i==np){ MC = C_h[np]; AC = 0; }
    f_MFHB(alpha,beta,i,f);

    double tau = 2.0*M_PI*i/(dlnk*np);
    double MCftau = MC * f[0], ACftau = AC + f[1] + M_LN2*tau;
    Cftau_h[2*np-i] = MCftau * sin(ACftau);
    Cftau_h[i] = MCftau * cos(ACftau);
  }

  bfft(Cftau_h,2*np); //now it's real

  //assemble J array
  double sl = (ell%2==0 ? 1 : -1);
  double pre = sl / (2.0*M_PI*M_PI*np*np);
  for(int i=0; i<np; i++){
    double ki=exp(lnk_pad_min+dlnk*i), k2_npm2=pow(ki*2,3.0+2.0*nu+alpha+beta);
    Ji[i] = pre * k2_npm2 * Cftau_h[2*i];
  }

  return 0;
}

const int tZ = 10; //number of Taylor expansion terms to keep
const double epsZ = 1e-2; //switch to expansion for r<epsZ or r>1/epsZ
 
double Zreg_n(int n, double r){

  if(n<0) return Zreg_n(-n,1.0/r);
  
  double Z = 0, lnkq = log(fabs((1.0+r)/(1.0-r)));

  switch(n){
  case 0:
    Z = 1.0;
    break;
  case 1:
    if(r < epsZ){
      for(int m=0; m<tZ; m++) Z += 2.0*pow(r,2.0*m+1.0)*(1.0-r) / (2.0*m+1.0); }
    else if(r > 1.0/epsZ){
      for(int m=0; m<tZ; m++) Z += 2.0*pow(r,-2.0*m-1.0)*(1.0-r)/(2.0*m+1.0); }
    else if(r == 1){ Z = 0.0; }
    else Z = (1.0 - r) * lnkq;
    break;
  case 2:
    if( r < epsZ){
      Z = 2.0*r;
      for(int m=0; m<tZ; m++) 
        Z += 2.0*pow(r,2.0*m+3.0) / ((2.0*m+1.0)*(2.0*m+3.0));
    }
    else if(r > 1.0/epsZ){
      for(int m=0; m<tZ; m++)
        Z += 2.0*pow(r,-2.0*m-1.0) / ((2.0*m+1.0)*(2.0*m+3.0));
    }
    else if(r == 1){ Z = 1.0; }
    else Z = r + 0.5*(1.0-r*r)*lnkq;
    break;
  case 3:
    if( r < epsZ){
      Z = r*r;
      for(int m=0; m<tZ; m++)
        Z += (1.0-cu(r))*pow(r,2*m+1) / (2.0*m+1.0);
    }
    else if(r > 1.0/epsZ){
      for(int m=0; m<tZ; m++)
        Z += pow(r,-2*m) *((2.0*m+3.0)/r-2.0*m-1.0) / ((2.0*m+1.0)*(2.0*m+3.0));
    }
    else if(r == 1){ Z = 1.0; }
    else Z = sq(r) + 0.5*(1.0-cu(r))*lnkq;
    break;
  case 4:
    if( r < epsZ){
      Z = (4.0/3.0) * (r + cu(r));
      for(int m=0; m<tZ; m++)
        Z += -4.0*pow(r,2*m+5) / ((2.0*m+1.0)*(2.0*m+5.0));
    }
    else if(r > 1.0/epsZ){
      for(int m=0; m<tZ; m++)
        Z += 4.0 * pow(r,-2*m-1) / ((2.0*m+1.0)*(2.0*m+5.0));
    }
    else if(r == 1){ Z = 4.0/3.0; }
    else Z = cu(r) + r/3.0 + 0.5*(1.0-sq(sq(r)))*lnkq;
    break;
  case 5:
    if( r < epsZ){
      Z = sq(sq(r)) + sq(r)/3.0;
      for(int m=0; m<tZ; m++)
        Z += (1.0-cu(r)*sq(r))*pow(r,2*m+1) / (2.0*m + 1.0);
    }
    else if(r > 1.0/epsZ){
      for(int m=0; m<tZ; m++)
        Z += pow(r,-2*m) *((2.0*m+5.0)/r-2.0*m-1.0) / ((2.0*m+1.0)*(2.0*m+5.0));
    }
    else if(r == 1){ Z = 4.0/3.0; }
    else Z = sq(sq(r)) + sq(r)/3.0 + 0.5*(1.0-cu(r)*sq(r))*lnkq;
    break;
  default:
    cout << "ERROR in Zreg_n for n=" << n << ": kernel terms only defined "
         << "for |n| <= 5.  Aborting." << endl;
    abort();
    break;
  }

  return Z;
}

int PZ_reg(int n, const double *Pq, const double *Pk, double *PZn){

  //for s[m]=log(q_m): Fs[m] = Pq(q_m) and Gs[m] = q_m^{-3}*Z(1/q_m)
  double Fs[4*np], Gs[4*np], FGconv[4*np];
  for(int i=0; i<4*np; i++){ Fs[i]=(i<np ? Pq[i] : 0);   Gs[i] = 0; }
  
  for(int i=0; i<np; i++){

    //r>1
    double si=dlnk*(i-np), r=exp(-si), r2=sq(r), r3=r*r2, r4=sq(r2), r5=r*r4;
    double Zi = Zreg_n(n,r);
    Gs[i] = Zi * r3;
  }

  Gs[np] = Zreg_n(n,1.0); //r=1   //12.0 + 10.0 + 100.0 - 42.0;

  for(int i=np+1; i<2*np; i++){
    //r<1
    double si=dlnk*(i-np), r=exp(-si), r2=sq(r), r3=r*r2, r4=sq(r2), r5=r*r4;
    double Zi = Zreg_n(n,r);
    Gs[i] = Zi * r3;
  }

  convolve_bruteforce(4*np,Fs,Gs,FGconv);
  //convolve(4*np,Fs,Gs,FGconv);

  //double pre = dlnk / (252.0 * sq(2.0*M_PI));
  double pre = dlnk / (2.0 * sq(M_PI));
  for(int i=0; i<np; i++){
    double lnk = lnk_pad_min + dlnk*i, k = exp(lnk), k3=k*k*k;
    double kfac = k3 * Pk[i];
    PZn[i] = pre * kfac * FGconv[i+np];
  }

  return 0;
}

const int nJ = 63, nJn0 = 63;

const int ell_n[7]   = {0, 0, 1, 2, 2, 3, 4};
const int alpha_n[7] = {0, 2, 1, 0, 2, 1, 0};

const int elln0_n[7]   = {0, 2, 4, 0, 2, 4, 6};
const int alphan0_n[7] = {0, 0, 0, 2, 2, 2, 2};
const int betan0_n[7]  = {2, 2, 2, 2, 2, 2, 2};

const int Z_n[7] = {0, 1, -1, 3, -3, 5, -5};

int compute_Aacdbef_Rlabc_PTjm_PMRn_full
( double eta, const double *y, double *Aacdbef, double Rlabc[nUQ*nk], 
  double PTjm[9][nk], double PMRn[8][nk]){
  
  //initialize to zero; only compute nonzero components
  for(int i=0; i<nUQ*nk; i++) Rlabc[i] = 0;
  for(int i=0; i<64*nk; i++) Aacdbef[i] = 0;
  for(int i=0; i<nk; i++){ PTjm[0][i] = 0; PTjm[1][i] = 0; PTjm[2][i] = 0; 
    PTjm[3][i] = 0; PTjm[4][i] = 0; PTjm[5][i] = 0; PTjm[6][i] = 0; 
    PTjm[7][i] = 0; PTjm[8][i] = 0; 
    PMRn[0][i] = 0; PMRn[1][i] = 0; PMRn[2][i] = 0; PMRn[3][i] = 0; 
    PMRn[4][i] = 0; PMRn[5][i] = 0; PMRn[6][i] = 0; PMRn[7][i] = 0;
  }
  
  //extrapolate power spectra.  P ~ k^{n_s} at low k, Eisenstein-Hu at high k
  double P[3*np];
  for(int i=0; i<np; i++){
    double k = exp(lnk_pad_min + dlnk*i), Win = WP(lnk_pad_min + dlnk*i);
    P[i] = Pab(0,0,k,y) * Win;
    P[i+np] = Pab(0,1,k,y) * Win;
    P[i+2*np] = Pab(1,1,k,y) * Win;
  }

  //full TRG calculation with input P; do improved 1-loop later.
  double J[nJ][np], Jn0[nJn0][np], PZ[nJ][np];

#pragma omp parallel for schedule(dynamic)
  for(int iJ=0; iJ<nJ; iJ++){
    int n = iJ/9, iabcd = iJ%9, iab = iabcd/3, icd = iabcd%3;
    J_MFHB(alpha_n[n],-alpha_n[n],ell_n[n], &P[iab*np],&P[icd*np], J[iJ]);
    //PZ_reg(Z_n[n], &P[iab*np],&P[icd*np], PZ[iJ]);
  }

  /**/
#pragma omp parallel for schedule(dynamic)
  for(int iJ=0; iJ<nJ; iJ+=3){
    int n = iJ/9, iabcd = iJ%9, iab = iabcd/3, icd = 0;
    PZ_reg(Z_n[n], &P[iab*np],&P[icd*np], PZ[iJ]);

    for(int i=0; i<np; i++){
      PZ[iJ+1][i] = PZ[iJ][i] * P[1*np+i] / (P[0*np+i] + 1e-100);
      PZ[iJ+2][i] = PZ[iJ][i] * P[2*np+i] / (P[0*np+i] + 1e-100);
    }
  }
  /**/

  if(C.PRINTRSD()){
#pragma omp parallel for schedule(dynamic)
    for(int iJ=0; iJ<nJn0; iJ++){
      int n = iJ/9, iabcd = iJ%9, iab = iabcd/3, icd = iabcd%3;
      J_MFHB(alphan0_n[n], betan0_n[n], elln0_n[n], 
	     &P[iab*np], &P[icd*np], Jn0[iJ]);
    }
  }

#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<nk; i++){
    double k = exp(lnkmin + dlnk*i), k2 = k*k, 
      pre_A = k / (4.0*M_PI), pre_R = 1.0/(2.0*M_PI*k);
    double Jterms = 0, PZterms = 0;

    //A_{acd,bef}
    Jterms = J[4*9+1][nshift+i]/6 
      +J[2*9+1][nshift+i]/2 
      +J[0*9+1][nshift+i]/4 
      +J[1*9+1][nshift+i]/12 
      + J[3*9+3][nshift+i]/6 
      + J[2*9+3][nshift+i]/4 
      + J[2*9+1][nshift+i]/4 
      + J[0*9+3][nshift+i]/3;
    PZterms = -PZ[0*9+1][nshift+i]/12.0 
      + (PZ[4*9+3][nshift+i] 
	 - PZ[2*9+3][nshift+i] 
	 + PZ[0*9+3][nshift+i] 
	 + PZ[1*9+3][nshift+i]/2
         - PZ[3*9+1][nshift+i] 
	 + PZ[1*9+1][nshift+i] 
	 + PZ[0*9+1][nshift+i]*3 
	 - PZ[2*9+1][nshift+i]/2) / 16;
    Aacdbef[8*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[4*9+2][nshift+i]/6 
      + J[2*9+2][nshift+i]/2 
      +J[0*9+2][nshift+i]/4 
      +J[1*9+2][nshift+i]/12
      + J[3*9+4][nshift+i]/6 
      + J[2*9+4][nshift+i]/4 
      + J[2*9+4][nshift+i]/4 
      + J[0*9+4][nshift+i]/3;
    PZterms = 0;
    Aacdbef[9*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[4*9+4][nshift+i]/6 
      +J[2*9+4][nshift+i]/2 
      +J[0*9+4][nshift+i]/4 
      +J[1*9+4][nshift+i]/12
      + J[3*9+6][nshift+i]/6 
      + J[2*9+6][nshift+i]/4 
      + J[2*9+2][nshift+i]/4 
      + J[0*9+6][nshift+i]/3;
    PZterms = -PZ[0*9+4][nshift+i]/12.0
      + (PZ[4*9+6][nshift+i] 
	 - PZ[2*9+6][nshift+i] 
	 + PZ[0*9+6][nshift+i] 
	 + PZ[1*9+6][nshift+i]/2
         - PZ[3*9+4][nshift+i] 
	 + PZ[1*9+4][nshift+i] 
	 + PZ[0*9+4][nshift+i]*3 
	 - PZ[2*9+4][nshift+i]/2) / 16;
    Aacdbef[10*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[4*9+5][nshift+i]/6 
      +J[2*9+5][nshift+i]/2 
      +J[0*9+5][nshift+i]/4 
      +J[1*9+5][nshift+i]/12
      + J[3*9+7][nshift+i]/6 
      + J[2*9+7][nshift+i]/4 
      + J[2*9+5][nshift+i]/4 
      + J[0*9+7][nshift+i]/3;
    PZterms = 0;
    Aacdbef[11*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[5*9+4][nshift+i]/5 
      + J[3*9+4][nshift+i]/2 
      + J[4*9+4][nshift+i]/6 
      + 0.55*J[2*9+4][nshift+i]
      + J[2*9+4][nshift+i]/4 
      +J[0*9+4][nshift+i]/4 
      + J[1*9+4][nshift+i]/12;
    PZterms = -PZ[0*9+2][nshift+i]/12.0
      + (PZ[4*9+4][nshift+i] 
	 - PZ[2*9+4][nshift+i] 
	 + PZ[0*9+4][nshift+i] 
	 + PZ[1*9+4][nshift+i]/2
         - PZ[3*9+2][nshift+i] 
	 + PZ[1*9+2][nshift+i] 
	 + PZ[0*9+2][nshift+i]*3 
	 - PZ[2*9+2][nshift+i]/2) / 16;
    Aacdbef[12*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[5*9+5][nshift+i]/5 
      + J[3*9+5][nshift+i]/2 
      + J[4*9+5][nshift+i]/6 
      + 0.55*J[2*9+5][nshift+i]
      + J[2*9+7][nshift+i]/4 
      +J[0*9+5][nshift+i]/4 
      + J[1*9+5][nshift+i]/12;
    PZterms = 0;
    Aacdbef[13*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[5*9+7][nshift+i]/5 
      + J[3*9+7][nshift+i]/2 
      + J[4*9+7][nshift+i]/6 
      + 0.55*J[2*9+7][nshift+i]
      + J[2*9+5][nshift+i]/4 
      +J[0*9+7][nshift+i]/4 
      + J[1*9+7][nshift+i]/12;
    PZterms = -PZ[0*9+5][nshift+i]/12.0
      + (PZ[4*9+7][nshift+i] 
	 - PZ[2*9+7][nshift+i] 
	 + PZ[0*9+7][nshift+i] 
	 + PZ[1*9+7][nshift+i]/2
         - PZ[3*9+5][nshift+i] 
	 + PZ[1*9+5][nshift+i] 
	 + PZ[0*9+5][nshift+i]*3 
	 - PZ[2*9+5][nshift+i]/2) / 16;
    Aacdbef[14*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[5*9+8][nshift+i]/5 
      + J[3*9+8][nshift+i]/2 
      + J[4*9+8][nshift+i]/6 
      + 0.55*J[2*9+8][nshift+i]
      + J[2*9+8][nshift+i]/4 
      +J[0*9+8][nshift+i]/4 
      + J[1*9+8][nshift+i]/12;
    PZterms = 0;
    Aacdbef[15*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = (J[5*9+1][nshift+i]/5 
	      + J[3*9+1][nshift+i]/2 
	      + J[4*9+1][nshift+i]/6 
	      + 0.55*J[2*9+1][nshift+i]
              + J[2*9+3][nshift+i]/4 
	      + J[0*9+1][nshift+i]/4 
	      + J[1*9+1][nshift+i]/12) * 2.0;
    PZterms = (-PZ[4*9+1][nshift+i]*2 
	       + PZ[2*9+1][nshift+i]*2 
	       - PZ[0*9+1][nshift+i]*2 
	       - PZ[1*9+1][nshift+i]
               + PZ[6*9+3][nshift+i]*2 
	       - PZ[4*9+3][nshift+i]*4 
	       + PZ[2*9+3][nshift+i]) / 16.0;
    Aacdbef[56*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[5*9+2][nshift+i]/5 
      + J[3*9+2][nshift+i]/2 
      + J[4*9+2][nshift+i]/6 
      + 0.55*J[2*9+2][nshift+i]
      + J[2*9+6][nshift+i]/4 
      + J[0*9+2][nshift+i]/4 
      + J[1*9+2][nshift+i]/12
      + J[5*9+4][nshift+i]/5 
      + J[3*9+4][nshift+i]/2 
      + J[4*9+4][nshift+i]/6 
      + 0.55*J[2*9+4][nshift+i]
      + J[2*9+4][nshift+i]/4 
      + J[0*9+4][nshift+i]/4 
      + J[1*9+4][nshift+i]/12;
    PZterms = (-PZ[4*9+4][nshift+i] 
	       + PZ[2*9+4][nshift+i] 
	       - PZ[0*9+4][nshift+i] 
	       - PZ[1*9+4][nshift+i]/2
               + PZ[6*9+6][nshift+i] 
	       - PZ[4*9+6][nshift+i]*2 
	       + PZ[2*9+6][nshift+i]/2) / 16.0;
    Aacdbef[57*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = (J[5*9+5][nshift+i]/5 
	      + J[3*9+5][nshift+i]/2 
	      + J[4*9+5][nshift+i]/6 
	      + 0.55*J[2*9+5][nshift+i]
              + J[2*9+7][nshift+i]/4 
	      + J[0*9+5][nshift+i]/4 
	      + J[1*9+5][nshift+i]/12) * 2.0;
    PZterms = 0;
    Aacdbef[59*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[6*9+4][nshift+i]*8/35 
      + 0.4*J[5*9+4][nshift+i] 
      + 0.4*J[5*9+4][nshift+i]
      + J[3*9+4][nshift+i]*19/21 
      + J[4*9+4][nshift+i]/6 
      + J[4*9+4][nshift+i]/6 
      + 0.6*J[2*9+4][nshift+i]
      + 0.6*J[2*9+4][nshift+i] 
      + J[0*9+4][nshift+i]*11/30 
      + J[1*9+4][nshift+i]/12 
      + J[1*9+4][nshift+i]/12;
    PZterms = (-PZ[4*9+2][nshift+i]*2 
	       + PZ[2*9+2][nshift+i]*2 
	       - PZ[0*9+2][nshift+i]*2 
	       - PZ[1*9+2][nshift+i]
               + PZ[6*9+4][nshift+i]*2 
	       - PZ[4*9+4][nshift+i]*4 
	       + PZ[2*9+4][nshift+i]) / 16.0;
    Aacdbef[60*nk+i] = pre_A * (Jterms + PZterms);

    Jterms = J[6*9+5][nshift+i]*8/35 
      + 0.4*J[5*9+5][nshift+i] 
      + 0.4*J[5*9+7][nshift+i]
      + J[3*9+5][nshift+i]*19/21 
      + J[4*9+5][nshift+i]/6 
      + J[4*9+7][nshift+i]/6 
      + 0.6*J[2*9+5][nshift+i]
      + 0.6*J[2*9+7][nshift+i] 
      + J[0*9+5][nshift+i]*11/30 
      + J[1*9+5][nshift+i]/12 
      + J[1*9+7][nshift+i]/12;
    PZterms = (-PZ[4*9+5][nshift+i] 
	       + PZ[2*9+5][nshift+i] 
	       - PZ[0*9+5][nshift+i] 
	       - PZ[1*9+5][nshift+i]/2
               + PZ[6*9+7][nshift+i] 
	       - PZ[4*9+7][nshift+i]*2 
	       + PZ[2*9+7][nshift+i]/2) / 16.0;
    Aacdbef[61*nk+i] = pre_A * (Jterms + PZterms);
    
    Jterms = J[6*9+8][nshift+i]*8/35 
      + 0.4*J[5*9+8][nshift+i] 
      + 0.4*J[5*9+8][nshift+i]
      + J[3*9+8][nshift+i]*19/21 
      + J[4*9+8][nshift+i]/6 
      + J[4*9+8][nshift+i]/6 
      + 0.6*J[2*9+8][nshift+i]
      + 0.6*J[2*9+8][nshift+i] 
      + J[0*9+8][nshift+i]*11/30 
      + J[1*9+8][nshift+i]/12 
      + J[1*9+8][nshift+i]/12;
    PZterms = 0;
    Aacdbef[63*nk+i] = pre_A * (Jterms + PZterms);

    //symmetries: A_{acd,bef} = A_{adc,bfe}
    Aacdbef[16*nk+i] = Aacdbef[8*nk+i];
    Aacdbef[18*nk+i] = Aacdbef[9*nk+i];
    Aacdbef[17*nk+i] = Aacdbef[10*nk+i];
    Aacdbef[19*nk+i] = Aacdbef[11*nk+i];
    Aacdbef[20*nk+i] = Aacdbef[12*nk+i];
    Aacdbef[22*nk+i] = Aacdbef[13*nk+i];
    Aacdbef[21*nk+i] = Aacdbef[14*nk+i];
    Aacdbef[23*nk+i] = Aacdbef[15*nk+i];
    Aacdbef[58*nk+i] = Aacdbef[57*nk+i];
    Aacdbef[62*nk+i] = Aacdbef[61*nk+i];


    //R^{ell}_{abc} 
    for(int a=0; a<2; a++){
      for(int b=0; b<2; b++){
        for(int c=0; c<2; c++){

          //ell = 1
          if(a==0){
            Jterms = 0.4 * J[9*5 + 3*b + c + 1][nshift+i]
              - 1.4 * J[9*2 + 3*b + c + 1][nshift+i]
              - J[9*2 + 3*c + b + 3][nshift+i]
              - 2.0 * J[9*0 + 3*b + c + 1][nshift+i]
              + 0.4 * J[9*5 + 3*c + b + 1][nshift+i]
              + (2.0/3.0) * J[9*3 + 3*b + c + 3][nshift+i]
              - (2.0/3.0) * J[9*4 + 3*c + b + 1][nshift+i]
              - 2.4 * J[9*2 + 3*c + b + 1][nshift+i]
              - (5.0/3.0) * J[9*0 + 3*b + c + 3][nshift+i]
              - (1.0/3.0) * J[9*1 + 3*c + b + 1][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }
          else{
            Jterms = (16.0/35.0) * J[9*6 + 3*b + c + 4][nshift+i]
              - 0.4 * J[9*5 + 3*c + b + 4][nshift+i]
              + 0.4 * J[9*5 + 3*b + c + 4][nshift+i]
              - (46.0/21.0) * J[9*3 + 3*b + c + 4][nshift+i]
              - (2.0/3.0) * J[9*4 + 3*b + c + 4][nshift+i]
              - 2.6 * J[9*2 + 3*c + b + 4][nshift+i]
              - 1.4 * J[9*2 + 3*b + c + 4][nshift+i]
              - (19.0/15.0) * J[9*0 + 3*b + c + 4][nshift+i]
              - (1.0/3.0) * J[9*1 + 3*c + b + 4][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }
          
          if(b==0){
            PZterms = -(13.0/12.0) * PZ[9*0 + 3*c + a + 1][nshift+i]
              + (5.0/16.0) * PZ[9*2 + 3*c + a + 1][nshift+i]
              - (7.0/16.0) * PZ[9*1 + 3*c + a + 1][nshift+i]
              - 0.125 * PZ[9*4 + 3*c + a + 1][nshift+i]
              + 0.375 * PZ[9*3 + 3*c + a + 1][nshift+i]
              - 0.375 * PZ[9*0 + 3*c + a + 3][nshift+i]
              + (7.0/16.0) * PZ[9*2 + 3*c + a + 3][nshift+i]
              - (3.0/16.0) * PZ[9*1 + 3*c + a + 3][nshift+i]
              - 0.625 * PZ[9*4 + 3*c + a + 3][nshift+i]
              + 0.125 * PZ[9*6 + 3*c + a + 3][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = -(1.0/3.0) * PZ[9*0 + 3*c + a + 4][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          
          if(c==0){
            PZterms = 0.125 * PZ[9*6 + 3*b + a + 3][nshift+i]
              - 0.375 * PZ[9*4 + 3*b + a + 3][nshift+i]
              + (3.0/16.0) * PZ[9*2 + 3*b + a + 3][nshift+i]
              - (1.0/16.0) * PZ[9*1 + 3*b + a + 3][nshift+i]
              - 0.125 * PZ[9*0 + 3*b + a + 3][nshift+i]
              - 0.125 * PZ[9*4 + 3*b + a + 1][nshift+i]
              + (3.0/16.0) * PZ[9*2 + 3*b + a + 1][nshift+i]
              - (3.0/16.0) * PZ[9*1 + 3*b + a + 1][nshift+i]
              + 0.125 * PZ[9*3 + 3*b + a + 1][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = (1.0/3.0) * PZ[9*0 + 3*b + a + 4][nshift+i];
            Rlabc[(8*0+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          
          //ell = 2
          if(a==0){
            Jterms = 0.6 * J[9*5 + 3*b + c + 1][nshift+i]
              + J[9*3 + 3*b + c + 1][nshift+i]
              - 0.6 * J[9*2 + 3*b + c + 1][nshift+i]
              - J[9*0 + 3*b + c + 1][nshift+i]
              + 0.6 * J[9*5 + 3*c + b + 1][nshift+i]
              + J[9*3 + 3*b + c + 3][nshift+i]
              - 0.6 * J[9*2 + 3*c + b + 1][nshift+i]
              - J[9*0 + 3*b + c + 3][nshift+i];
            Rlabc[(8*1+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }
          else{
            Jterms = 24.0/35.0 * J[9*6 + 3*b + c + 4][nshift+i]
              - 1.0 * J[9*5 + 3*c + b + 4][nshift+i]
              + 2.2 * J[9*5 + 3*b + c + 4][nshift+i]
              - (2.0/7.0) * J[9*3 + 3*b + c + 4][nshift+i]
              - 0.6 * J[9*2 + 3*b + c + 4][nshift+i]
              - 0.6 * J[9*2 + 3*c + b + 4][nshift+i]
              - 0.4 * J[9*0 + 3*b + c + 4][nshift+i];
            Rlabc[(8*1+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }

          if(b==0){
            PZterms = -(1.0/2.0) * PZ[9*0 + 3*c + a + 1][nshift+i]
              + (9.0/32.0) * PZ[9*2 + 3*c + a + 1][nshift+i]
              - (9.0/32.0) * PZ[9*1 + 3*c + a + 1][nshift+i]
              - (3.0/16.0) * PZ[9*4 + 3*c + a + 1][nshift+i]
              + (3.0/16.0) * PZ[9*3 + 3*c + a + 1][nshift+i]
              - (3.0/16.0) * PZ[9*0 + 3*c + a + 3][nshift+i]
              - (3.0/32.0) * PZ[9*1 + 3*c + a + 3][nshift+i]
              + (9.0/32.0) * PZ[9*2 + 3*c + a + 3][nshift+i]
              - (9.0/16.0) * PZ[9*4 + 3*c + a + 3][nshift+i]
              + (3.0/16.0) * PZ[9*6 + 3*c + a + 3][nshift+i];
            Rlabc[(8*1+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = 0;
            Rlabc[(8*1+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }

          if(c==0){
            PZterms = (3.0/16.0) * PZ[9*6 + 3*b + a + 3][nshift+i]
              - (9.0/16.0) * PZ[9*4 + 3*b + a + 3][nshift+i]
              + (9.0/32.0) * PZ[9*2 + 3*b + a + 3][nshift+i]
              - (3.0/32.0) * PZ[9*1 + 3*b + a + 3][nshift+i]
              - (3.0/16.0) * PZ[9*0 + 3*b + a + 3][nshift+i]
              + (3.0/16.0) * PZ[9*3 + 3*b + a + 1][nshift+i]
              - (3.0/16.0) * PZ[9*4 + 3*b + a + 1][nshift+i]
              - (9.0/32.0) * PZ[9*1 + 3*b + a + 1][nshift+i]
              + (9.0/32.0) * PZ[9*2 + 3*b + a + 1][nshift+i]
              - (1.0/2.0) * PZ[9*0 + 3*b + a + 1][nshift+i];
            Rlabc[(8*1+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = 0;
            Rlabc[(8*1+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }

          //ell = 3
          if(a==0){
            Jterms = (
                      (4.0/7.0) * Jn0[9*2 + 3*c + b + 3][nshift+i]
                      - (40.0/21.0) * Jn0[9*1 + 3*c + b + 3][nshift+i]
                      + (4.0/3.0) * Jn0[9*0 + 3*c + b + 3][nshift+i]
                      - (4.0/7.0) * Jn0[9*2 + 3*b + c + 3][nshift+i]
                      + (40.0/21.0) * Jn0[9*1 + 3*b + c + 3][nshift+i]
                      - (4.0/3.0) * Jn0[9*0 + 3*b + c + 3][nshift+i]
                      ) / k2
              - J[9*5 + 3*b + c + 1][nshift+i]
              + J[9*2 + 3*b + c + 1][nshift+i]
              - (5.0/3.0) * J[9*3 + 3*b + c + 3][nshift+i]
              + (5.0/3.0) * J[9*0 + 3*b + c + 3][nshift+i];
            Rlabc[(8*2+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }
          else{
            Jterms = -(4.0/7.0) * J[9*6 + 3*b + c + 4][nshift+i]
              - J[9*5 + 3*b + c + 4][nshift+i]
              + (5.0/21.0) * J[9*3 + 3*b + c + 4][nshift+i]
              + J[9*2 + 3*b + c + 4][nshift+i]
              + (1.0/3.0) * J[9*0 + 3*b + c + 4][nshift+i];
            Rlabc[(8*2+4*a+2*b+c)*nk + i] = pre_R * Jterms;
          }
          
          if(b==0){
            PZterms = (35.0/32.0) * PZ[9*0 + 3*c + a + 1][nshift+i]
              + (5.0/32.0) * PZ[9*5 + 3*c + a + 1][nshift+i]
              - (5.0/8.0) * PZ[9*3 + 3*c + a + 1][nshift+i]
              + (5.0/32.0) * PZ[9*4 + 3*c + a + 1][nshift+i]
              - (5.0/16.0) * PZ[9*2 + 3*c + a + 1][nshift+i]
              + (15.0/32.0) * PZ[9*1 + 3*c + a + 1][nshift+i]
              + (55.0/96.0) * PZ[9*0 + 3*c + a + 3][nshift+i]
              - (5.0/32.0) * PZ[9*6 + 3*c + a + 3][nshift+i]
              + (5.0/8.0) * PZ[9*4 + 3*c + a + 3][nshift+i]
              - (5.0/32.0) * PZ[9*3 + 3*c + a + 3][nshift+i]
              - (15.0/32.0) * PZ[9*2 + 3*c + a + 3][nshift+i]
              + (5.0/16.0) * PZ[9*1 + 3*c + a + 3][nshift+i];
            Rlabc[(8*2+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = (1.0/3.0) * PZ[9*0 + 3*c + a + 4][nshift+i];
            Rlabc[(8*2+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }

          if(c==0){
            PZterms = 1.25 * (
                              -0.125 * PZ[9*6 + 3*b + a + 3][nshift+i]
                              + 0.25 * PZ[9*4 + 3*b + a + 3][nshift+i]
                              - (5.0/24.0) * PZ[9*0 + 3*b + a + 3][nshift+i]
                              - 0.125 * PZ[9*1 + 3*b + a + 3][nshift+i]
                              + 0.125 * PZ[9*3 + 3*b + a + 3][nshift+i]
                              - 0.125 * PZ[9*5 + 3*b + a + 1][nshift+i]
                              + 0.25 * PZ[9*3 + 3*b + a + 1][nshift+i]
                              - (5.0/24.0) * PZ[9*0 + 3*b + a + 1][nshift+i]
                              - 0.125 * PZ[9*2 + 3*b + a + 1][nshift+i]
                              + 0.125 * PZ[9*4 + 3*b + a + 1][nshift+i]
                              );
            Rlabc[(8*2+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }
          else{
            PZterms = -(1.0/3.0) * PZ[9*0 + 3*b + a + 4][nshift+i];
            Rlabc[(8*2+4*a+2*b+c)*nk + i] += pre_R * PZterms;
          }

        }
      }
    }


    //P_{T,jm}[index][wavenumber] is the term multiplying mu^j b^m
    //allowed values of index, j, and m
    //i = 0 1 2 3 4 5 6 7 8
    //j = 2 2 2 4 4 4 6 6 8
    //m = 2 1 0 2 1 0 1 0 0
    PTjm[0][i] = (1.0/3.0) * J[9*3 + 3*0 + 0 + 4][nshift+i]
      - (1.0/3.0) * J[9*0 + 3*0 + 0 + 4][nshift+i];
    PTjm[1][i] = 2.0 * (
                        (-3.0/35.0) * Jn0[9*2 + 3*1 + 0 + 4][nshift+i]
                        + (2.0/7.0) * Jn0[9*1 + 3*1 + 0 + 4][nshift+i]
                        - 0.2 * Jn0[9*0 + 3*1 + 0 + 4][nshift+i]
                        ) / k2;

    PTjm[2][i] = ( (5.0/231.0) * Jn0[9*6 + 3*1 + 1 + 4][nshift+i]
                   - (9.0/77.0) * Jn0[9*5 + 3*1 + 1 + 4][nshift+i]
                   + (5.0/21.0) * Jn0[9*4 + 3*1 + 1 + 4][nshift+i]
                   - (1.0/7.0) * Jn0[9*3 + 3*1 + 1 + 4][nshift+i]
                   ) / (k2*k2);

    PTjm[3][i] = (1.0/3.0) * J[9*3 + 3*0 + 0 + 4][nshift+i]
      + 2.0 * J[9*2 + 3*0 + 0 + 4][nshift+i]
      + (5.0/3.0) * J[9*0 + 3*0 + 0 + 4][nshift+i];

    PTjm[4][i] = -(6.0/5.0) * J[9*5 + 3*0 + 1 + 4][nshift+i]
      + 2.0 * J[9*3 + 3*1 + 0 + 4][nshift+i]
      + (6.0/5.0) * J[9*2 + 3*0 + 1 + 4][nshift+i]
      - 2.0 * J[9*0 + 3*1 + 0 + 4][nshift+i]
      + (
         (12.0/7.0) * Jn0[9*2 + 3*1 + 0 + 4][nshift+i]
         - (40.0/7.0) * Jn0[9*1 + 3*1 + 0 + 4][nshift+i]
         + 4.0 * Jn0[9*0 + 3*1 + 0 + 4][nshift+i]
         ) / k2;
    PTjm[5][i] = ( -(5.0/11.0) * Jn0[9*6 + 3*1 + 1 + 4][nshift+i]
                   + (27.0/11.0) * Jn0[9*5 + 3*1 + 1 + 4][nshift+i]
                   - 5.0 * Jn0[9*4 + 3*1 + 1 + 4][nshift+i]
                   + 3.0 * Jn0[9*3 + 3*1 + 1 + 4][nshift+i]
                   ) / (k2*k2)
      + (
         -(9.0/7.0) * Jn0[9*2 + 3*1 + 1 + 4][nshift+i]
         + (30.0/7.0) * Jn0[9*1 + 3*1 + 1 + 4][nshift+i]
         - 3.0 * Jn0[9*0 + 3*1 + 1 + 4][nshift+i]
         ) / k2
      + (27.0/70.0) * J[9*6 + 3*1 + 1 + 4][nshift+i]
      - (9.0/7.0) * J[9*3 + 3*1 + 1 + 4][nshift+i]
      + (9.0/10.0) * J[9*0 + 3*1 + 1 + 4][nshift+i];

    PTjm[6][i] = (-2.0 * Jn0[9*2 + 3*1 + 0 + 4][nshift+i]
                  + (20.0/3.0) * Jn0[9*1 + 3*1 + 0 + 4][nshift+i]
                  - (14.0/3.0) * Jn0[9*0 + 3*1 + 0 + 4][nshift+i]
                  ) / k2
      + 2.0 * J[9*5 + 3*0 + 1 + 4][nshift+i]
      - (2.0/3.0) * J[9*3 + 3*1 + 0 + 4][nshift+i]
      + 2.0 * J[9*2 + 3*1 + 0 + 4][nshift+i]
      + (14.0/3.0) * J[9*0 + 3*1 + 0 + 4][nshift+i];
    
    PTjm[7][i] = ( (15.0/11.0) * Jn0[9*6 + 3*1 + 1 + 4][nshift+i]
                   - (81.0/11.0) * Jn0[9*5 + 3*1 + 1 + 4][nshift+i]
                   + 15.0 * Jn0[9*4 + 3*1 + 1 + 4][nshift+i]
                   - 9.0 * Jn0[9*3 + 3*1 + 1 + 4][nshift+i]
                   ) / (k2*k2)
      + ( 6.0 * Jn0[9*2 + 3*1 + 1 + 4][nshift+i]
          - 20.0 * Jn0[9*1 + 3*1 + 1 + 4][nshift+i]
          + 14.0 * Jn0[9*0 + 3*1 + 1 + 4][nshift+i]
          ) / k2
      - (39.0/35.0) * J[9*6 + 3*1 + 1 + 4][nshift+i]
      - (6.0/5.0) * J[9*5 + 3*1 + 1 + 4][nshift+i]
      + (47.0/7.0) * J[9*3 + 3*1 + 1 + 4][nshift+i]
      + (6.0/5.0) * J[9*2 + 3*1 + 1 + 4][nshift+i]
      - (28.0/5.0) * J[9*0 + 3*1 + 1 + 4][nshift+i];

    PTjm[8][i] = ( -1.0 * Jn0[9*6 + 3*1 + 1 + 4][nshift+i]
                   + (27.0/5.0) * Jn0[9*5 + 3*1 + 1 + 4][nshift+i]
                   - 11.0 * Jn0[9*4 + 3*1 + 1 + 4][nshift+i]
                   + (33.0/5.0) * Jn0[9*3 + 3*1 + 1 + 4][nshift+i]
                   ) / (k2*k2)
      + (
         -(27.0/5.0) * Jn0[9*2 + 3*1 + 1 + 4][nshift+i]
         + 18.0 * Jn0[9*1 + 3*1 + 1 + 4][nshift+i]
         - (63.0/5.0) * Jn0[9*0 + 3*1 + 1 + 4][nshift+i]
         ) / k2
      + (59.0/70.0) * J[9*6 + 3*1 + 1 + 4][nshift+i]
      + 2.0 * J[9*5 + 3*1 + 1 + 4][nshift+i]
      - (36.0/7.0) * J[9*3 + 3*1 + 1 + 4][nshift+i]
      + (63.0/10.0) * J[9*0 + 3*1 + 1 + 4][nshift+i];


    //P_{MR,n}(k) are the bias correction integrals computed in 
    //McDonald and Roy JCAP08 (2009) 020 [arXiv:0902.0991].
    //Power spectrum indices a=0, b=0 used for all terms.
    //  n=0: delta^2,delta     n=4: delta^2,delta^2
    //  n=1: delta^2,theta     n=5: delta^2,s^2
    //  n=2: s^2,delta         n=6: s^2,s^2
    //  n=3: s^2,theta         n=7: 3nl
    const int nloMR = nshift - nk/2; //k_nloMR used for low-k limit
    PMRn[0][i] = (4.0/21.0) * J[9*3][nshift+i]  
      + J[9*2][nshift+i]
      + (17.0/21.0) * J[9*0][nshift+i];
    PMRn[1][i] = (8.0/21.0) * J[9*3][nshift+i]
      + J[9*2][nshift+i]
      + (13.0/21.0) * J[9*0][nshift+i];
    PMRn[2][i] = (16.0/245.0) * J[9*6][nshift+i]
      + (2.0/5.0) * J[9*5][nshift+i]
      + (254.0/441.0) * J[9*3][nshift+i]
      + (4.0/15.0) * J[9*2][nshift+i]
      + (8.0/315.0) * J[9*0][nshift+i];
    PMRn[3][i] = (32.0/245.0) * J[9*6][nshift+i]
      + (2.0/5.0) * J[9*5][nshift+i]
      + (214.0/441.0) *J[9*3][nshift+i]
      + (4.0/15.0) * J[9*2][nshift+i]
      + (16.0/315.0) * J[9*0][nshift+i];
    PMRn[4][i] = 0.5 * J[9*0][nshift+i] - 0.5*J[9*0][nloMR];
    PMRn[5][i] = ( J[9*3][nshift+i] - J[9*0][nloMR] ) / 3.0;
    PMRn[6][i] = (4.0/35.0)  * J[9*6][nshift+i]
      + (4.0/63.0) * J[9*3][nshift+i]
      + (2.0/45.0) * J[9*0][nshift+i]
      - (2.0/9.0) * J[9*0][nloMR];
    PMRn[7][i] = 0.5 * ( 
			(-15.0/128.0) * PZ[9*6][nshift+i]
			+ (15.0/32.0) * PZ[9*4][nshift+i]
			- (15.0/128.0) * PZ[9*3][nshift+i]
			- (45.0/128.0) * PZ[9*2][nshift+i]
			+ (15.0/64.0) * PZ[9*1][nshift+i]
			+ (55.0/128.0) * PZ[9*0][nshift+i]
			 );
  }
  
  return 0;
}

//1-loop computation: z1l at which to evaluate 1-loop
const double z1l = 10.0, eta_z1l = log((1.0+C.z_in())/(1.0+z1l));

int compute_Aacdbef_Rlabc_PTjm_PMRn_1loop
( double eta, const double *y, 
  double *Aacdbef, double Rlabc[nUQ*nk], 
  double PTjm[9][nk], double PMRn[8][nk]){
  
  static int init = 0;
  static double A_z1l[64*nk], R_z1l[nUQ*nk], PT_z1l[9][nk], PMR_z1l[8][nk], 
    D_z1l[nk];
  
  if(!init){

    //construct linear power spectrum at redshift z1l
    double y_z1l[3*nk+10], Dtemp[2];
    for(int i=0; i<nk; i++){
      C.D_dD(z1l,kArr[i],Dtemp);
      D_z1l[i] = Dtemp[0];
      double P_z1l = C.Plin_cb(z1l,kArr[i]);
      y_z1l[i] = log(P_z1l);
      y_z1l[nk+i] = y_z1l[i];  //log(P_z1l * f_z1l[i]);
      y_z1l[2*nk+i] = y_z1l[i];  //log(P_z1l * f_z1l[i]*f_z1l[i]);
    }

    //compute z=z1l values
    compute_Aacdbef_Rlabc_PTjm_PMRn_full(eta_z1l, y_z1l, 
					 A_z1l, R_z1l, PT_z1l, PMR_z1l);

    init = 1;
  }

  //scale outputs to proper redshift
  for(int i=0; i<nk; i++){
    double Dz[2], z = exp(-eta)*(1.0+C.z_in()) - 1;
    C.D_dD(z, kArr[i], Dz);
    double fz = Dz[1] / (Dz[0] * (1.0+z));
    double pre = pow(Dz[0]/D_z1l[i],4) * exp(-4.0*eta);

    for(int j=0; j<64; j++){ 
      int bef = j%8, b=bef/4, e=(bef%4)/2, f=bef%2;
      Aacdbef[j*nk+i] = pre * pow(fz,b+e+f+1) * A_z1l[j*nk+i];
    }

    for(int j=0; j<nUQ; j++){
      int abc = j%8, a=abc/4, b=(abc%4)/2, c=abc%2;
      Rlabc[j*nk+i] = pre * pow(fz,a+b+c+1) * R_z1l[j*nk+i];
    }

    for(int n=0; n<9; n++)
      PTjm[n][i] = pre * pow(fz,4-m_n[n]) * PT_z1l[n][i];

    for(int n=0; n<8; n++)
      PMRn[n][i] = pre * PMR_z1l[n][i];

  }

  return 0;
}

//when no bias is being applied, we can collapse all the m values for each j
int compute_Aacdbef_Rlabc_PTj
( double eta, const double *y,
  double *Aacdbef,
  double *Rlabc,
  double *PT2, double *PT4, double *PT6, double *PT8){

  double PTjm[9][nk], PMRn[8][nk];
  if(C.SWITCH_1LOOP()) 
    compute_Aacdbef_Rlabc_PTjm_PMRn_1loop(eta,y,Aacdbef,Rlabc,PTjm,PMRn);
  else compute_Aacdbef_Rlabc_PTjm_PMRn_full(eta,y,Aacdbef,Rlabc,PTjm,PMRn);

  for(int i=0; i<nk; i++){
    PT2[i] = PTjm[0][i] + PTjm[1][i] + PTjm[2][i];
    PT4[i] = PTjm[3][i] + PTjm[4][i] + PTjm[5][i];
    PT6[i] = PTjm[6][i] + PTjm[7][i];
    PT8[i] = PTjm[8][i];
  }

  return 0;
}

//same as above, but forces recomputation rather than using 1-loop
int compute_Aacdbef_Rlabc_PTj_full
( double eta, const double *y,
  double *Aacdbef,
  double *Rlabc,
  double *PT2, double *PT4, double *PT6, double *PT8){

  double PTjm[9][nk], PMRn[8][nk];
  compute_Aacdbef_Rlabc_PTjm_PMRn_full(eta,y,Aacdbef,Rlabc,PTjm,PMRn);

  for(int i=0; i<nk; i++){
    PT2[i] = PTjm[0][i] + PTjm[1][i] + PTjm[2][i];
    PT4[i] = PTjm[3][i] + PTjm[4][i] + PTjm[5][i];
    PT6[i] = PTjm[6][i] + PTjm[7][i];
    PT8[i] = PTjm[8][i];
  }

  return 0;
}

//frontends for parallel computation of A_{acd,bef}, R^\ell_{abc}, 
//P_{T,jm}, P_{MR,n}.  Figure out which function to use.
int compute_Aacdbef_Rlabc_PTjm_PMRn
( double eta, const double *y,
  double *Aacdbef, double Rlabc[nUQ*nk],
  double PTjm[9][nk], double PMRn[8][nk]){

  if(C.SWITCH_1LOOP())
    return compute_Aacdbef_Rlabc_PTjm_PMRn_1loop(eta,y,Aacdbef,Rlabc,PTjm,PMRn);

  return compute_Aacdbef_Rlabc_PTjm_PMRn_full(eta,y,Aacdbef,Rlabc,PTjm,PMRn);
}

///////////////////////////////////////////////////////////////////////////////
//Omega matrix (linear evolution)
double Omega(int i, int j, double A, double k){

  switch(2*i + j){

  case 0: //row 0, column 0
    return 1;
    break;

  case 1: //row 0, column 1
    return -1;
    break;

  case 2: //row 1, column 0
    return -1.5*C.Omega_m() * (C.f_cb()+C.Beta_P(A,k)) / (A*A*A * C.H2_H02(A));
    break;

  case 3: //row 1, column 1
    return 3.0 + C.dlnH_dlna(A);
    break;

  default:
    cout << "ERROR: Invalid option in Omega(int,int)." << endl;
    abort();
    break;
  }

  return exp(1000); //shouldn't ever get here
}

///////////////////////////////////////////////////////////////////////////////
//derivatives of power spectrum and I matrix
int derivatives_LIN(double eta, const double y[], double dy[], void *params){

  for(int i=0; i<nU*nk+nk; i++) dy[i]=0;
  double A = C.a_in() * exp(eta), z = 1.0/A - 1.0;

#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<nk; i++){
    double dPab_i[3]={0,0,0}, Pab_i[3]={exp(y[i]),exp(y[nk+i]),exp(y[2*nk+i])};
    for(int c=0; c<2; c++){ 
      dPab_i[0] -= Omega(0,c,A,kArr[i])*Pab_i[c] 
	+ Omega(0,c,A,kArr[i])*Pab_i[c]; //00
      dPab_i[1] -= Omega(0,c,A,kArr[i])*Pab_i[c+1] 
	+ Omega(1,c,A,kArr[i])*Pab_i[c]; //01=10
      dPab_i[2] -= Omega(1,c,A,kArr[i])*Pab_i[c+1] 
	+ Omega(1,c,A,kArr[i])*Pab_i[c+1]; //11
    }  
    dy[i] = dPab_i[0] / Pab_i[0];
    dy[nk+i] = dPab_i[1] / Pab_i[1];
    dy[2*nk+i] = dPab_i[2] / Pab_i[2];
  }

  return GSL_SUCCESS;
}

int derivatives(double eta, const double y[], double dy[], void *params){
  //the array y[nU*nk+nk] contains the power spectra and the I matrix:
  //  y[0..nk-1] is ln(P_00) from kArr[0] to kArr[nk-1]
  //  y[nk..2*nk-1] is ln(P_10) = ln(P_01) from kArr[0] to kArr[nk-1]
  //  y[2*nk..3*nk-1] is ln(P_11) from kArr[0] to kArr[nk-1]
  //  y[3*nk...(3+nI)*nk-1] is the 14 unique components of I
  //                             from kArr[0] to kArr[nk-1]
  //  ...
  //  y[nU*nk...nU*nk+nk-1] is the nk \ell values at which Clpp
  //                             is computed (using time-rg or halofit)
  
  if(DEBUG_ALL || DEBUG_INTEGRATION)
    cout << "#derivatives: start. a_in=" << C.a_in() 
	 << ", a=" << C.a_in()*exp(eta) << ", eta=" << eta << endl;

  //scale factor and red shift; note A=scale factor, a is an iteger
  double A = C.a_in()*exp(eta), z = 1.0/A - 1.0; 

  //initialize
  for(int i=0; i<nU*nk+C.n_phi()*nk; i++) dy[i]=0;
  
  //calculate Aacdbef
  double Aacdbef[64*nk], Iacdbef[64*nk], Rlabc[nUQ*nk], PT[4][nk];
  if(C.SWITCH_NONLINEAR() && z>=C.z_phi_trg_min()){
    compute_Aacdbef_Rlabc_PTj(eta, y, Aacdbef, Rlabc,
			      PT[0], PT[1], PT[2], PT[3]);
    I64(&y[nUP*nk],Iacdbef);
  }

  //iterate over k array
#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<nk; i++){

    //0..3*nk-1: derivatives of ln(P)
    double dPab_i[3]={0,0,0}, Pab_i[3]={exp(y[i]),exp(y[nk+i]),exp(y[2*nk+i])};
    for(int c=0; c<2; c++){
      dPab_i[0] -= Omega(0,c,A,kArr[i])*Pab_i[c] 
	+ Omega(0,c,A,kArr[i])*Pab_i[c]; //00
      dPab_i[1] -= Omega(0,c,A,kArr[i])*Pab_i[c+1] 
	+ Omega(1,c,A,kArr[i])*Pab_i[c]; //01=10
      dPab_i[2] -= Omega(1,c,A,kArr[i])*Pab_i[c+1] 
	+ Omega(1,c,A,kArr[i])*Pab_i[c+1]; //11

      if(C.SWITCH_NONLINEAR() && z>=C.z_phi_trg_min()){
	for(int d=0; d<2; d++){
	  int a=0, b=0, J0 = nAI(a,c,d,b,c,d), J1 = nAI(b,c,d,a,c,d);
	  dPab_i[0] += exp(eta)*4.0*M_PI/kArr[i]
	    * (Iacdbef[J0*nk+i] + Iacdbef[J1*nk+i]);

	  a=1; b=0; J0 = nAI(a,c,d,b,c,d); J1 = nAI(b,c,d,a,c,d);
	  dPab_i[1] += exp(eta)*4.0*M_PI/kArr[i]
            * (Iacdbef[J0*nk+i] + Iacdbef[J1*nk+i]);

	  a=1; b=1; J0 = nAI(a,c,d,b,c,d); J1 = nAI(b,c,d,a,c,d);
          dPab_i[2] += exp(eta)*4.0*M_PI/kArr[i]
            * (Iacdbef[J0*nk+i] + Iacdbef[J1*nk+i]);
	}                                                         
      }
    }

    dy[i] = dPab_i[0] / Pab_i[0];
    dy[nk+i] = dPab_i[1] / Pab_i[1];
    dy[2*nk+i] = dPab_i[2] / Pab_i[2];

    //instability in linear evolution as P_11 --> 0
    for(int jP=0; jP<nUP; jP++){
      if(dy[jP*nk+i] < -10.0) dy[jP*nk+i] = -10.0;
      if(dy[jP*nk+i] >  10.0) dy[jP*nk+i] =  10.0;
    }
    
    if(C.SWITCH_NONLINEAR() && z>=C.z_phi_trg_min()){

      //3*nk to 17*nk-1: derivatives of I
      for(int j=0; j<nUI; j++){
	dy[(j+nUP)*nk+i] = 2.0*exp(eta)*Aacdbef[JU[j]*nk + i];

	for(int g=0; g<2; g++){
	  int J1=nAI(aU[j],cU[j],dU[j],g,eU[j],fU[j]), 
	    J2=nAI(aU[j],cU[j],dU[j],bU[j],g,fU[j]),
	    J3=nAI(aU[j],cU[j],dU[j],bU[j],eU[j],g);
	  dy[(j+nUP)*nk+i] += -Omega(bU[j],g,A,kArr[i])*Iacdbef[J1*nk+i]
	    - Omega(eU[j],g,A,kArr[i])*Iacdbef[J2*nk+i]
	    - Omega(fU[j],g,A,kArr[i])*Iacdbef[J3*nk+i];
	}
      }

      //17*nk to 64*nk: beta0_abc(k)
      if(PRINTQ || C.PRINTRSD()){
	for(int num_ell=0; num_ell<nELL; num_ell++){
	  double Qlabc_i[8];
	  for(int j=0; j<8; j++){
	    Qlabc_i[j] = y[(nUP + nUI + num_ell*8 + j)*nk + i];
	    dy[(nUP + nUI + num_ell*8 + j)*nk+i] = 
	      2.0*exp(eta)*Rlabc[(num_ell*8+j)*nk+i];
	  }
	  
	  for(int a=0; a<2; a++){
	    for(int b=0; b<2; b++){
	      for(int c=0; c<2; c++){
		int j = 4*a + 2*b + c;
		for(int d=0; d<2; d++){
		  dy[(nUP + nUI + num_ell*8 + j)*nk+i] += 
		    -Omega(a,d,A,kArr[i])*Qlabc_i[4*d+2*b+c]
		    - Omega(b,d,A,kArr[i])*Qlabc_i[4*a+2*d+c]
		    - Omega(c,d,A,kArr[i])*Qlabc_i[4*a+2*b+d];
		}
	      }
	    }
	  }
	}	
      }
    }

  }

  //lensing potential power spectrum integration
  if(PRINTCLPP && A < 0.99999){

    //non-linear power interpolation
    tabulated_function lnPcbNL(nk,lnkArr,y), lnPcbHF;
    double lnPcbHF_interp[nk];
    if(C.use_Clpp_hft()){ 
      AU_halofit(A,lnPcbHF_interp);
      lnPcbHF.initialize(nk,lnkArr,lnPcbHF_interp);
    }

    double eta_star = -log((1.0+z_star)*C.a_in());
    double H0chi_star = C.H0chi(eta_star), H0chi = C.H0chi(eta);
    double H0_ratio = C.H_H0(A) / C.H_H0(10.0/13.0); //E(z)/E(0.3)
    double g_chi = 2.0 * (1.0 - H0chi/H0chi_star), H_H0 = C.H_H0(A);
    double pre_kap = 9.0 * C.Omega_m()*C.Omega_m() * H0h*H0h*H0h * g_chi*g_chi
      / (4.0 * A*A*A * H_H0);
    
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i<nk; i++){
      double ell = ellArr[i], k = H0h * ell / H0chi, ell4 = ell*ell*ell*ell;
      double sqrtPnu = sqrt(C.Plin_nu(z,k));
      
      //lensing potential power spectrum
      double PmHF=0, PmMT4=0, PmEE2=0;
      double sqrtPcbTRG = (A/C.a_in()) * exp(0.5*lnPcbNL(log(k))),
        sqrtPmTRG = C.f_cb()*sqrtPcbTRG + C.f_nu()*sqrtPnu,
        PmTRG = sqrtPmTRG*sqrtPmTRG;
      if(C.use_Clpp_hft()){
        double sqrtPcbHF = exp(0.5*lnPcbHF(log(k))),
	  sqrtPmHF = C.f_cb()*sqrtPcbHF + C.f_nu()*sqrtPnu;
	PmHF = sqrtPmHF*sqrtPmHF;
      }
      if(C.use_Clpp_mt4()) PmMT4 = Pm_MT4(eta,k);
      if(C.use_Clpp_ee2()) PmEE2 = Pm_EE2(eta,k,A_s_extern);

      for(int jkap=0; jkap<C.n_phi(); jkap++){
	if(z>=C.z_phi_min(jkap) && z<=C.z_phi_max(jkap)){

	  double Pm_phi = 0;
	  
	  switch((C.SWITCH_NONLINEAR()>0)*C.SWITCH_PHI_MAT(jkap)){
	  case 0: Pm_phi = C.Plin(z,k);                             break;
	  case 1: Pm_phi = PmTRG;                                   break;
	  case 2: Pm_phi = (z > C.z_phi_trg(jkap) ? PmTRG : PmHF);  break;
	  case 3: Pm_phi = (z > C.z_phi_trg(jkap) ? PmTRG : PmMT4); break;
          case 4: Pm_phi = (z > C.z_phi_trg(jkap) ? PmTRG : PmEE2); break;
	  default: Pm_phi = C.Plin(z,k);                            break;
	  }
	  
          double ph[] = { C.par_phi_p(jkap), C.par_phi_q(jkap),
                          C.par_phi_r(jkap), H0_ratio};  
	  //double baryon_sup = (C.SWITCH_BARYON_FEEDBACK(jkap)
	  //		       ? hyd_sup(z,k,C.par_phi_p(jkap)) : 1);
          double baryon_sup = hyd_sup(z,k,C.SWITCH_BARYON_FEEDBACK(jkap),ph);

	  dy[nU*nk + jkap*nk + i] = pre_kap/ell4 * Pm_phi * baryon_sup;
	}
      }
    }//end for i (Clpp multipoles)

    //DEBUG: print power spectra
    if(DEBUG_CONVERGENCE){
      for(int i=0; i<nk; i++){//assume nellkap == nk
        double a2_ain2 = A*A / (C.a_in()*C.a_in());
        cout << "#derivatives:DEBUG:k,Plin,Ptrg,Phf: " << kArr[i] << " "
  	     << C.Plin_cb(z,kArr[i]) << " " << a2_ain2*exp(y[i])
	     << " " << exp(lnPcbHF(lnkArr[i])) << endl;
      }
      cout << "#derivatives:DEBUG:k,Plin,Ptrg,Phf: " << endl
  	   << "#derivatives:DEBUG:k,Plin,Ptrg,Phf: " << endl;
    }

  }
    
  if(DEBUG_ALL || DEBUG_INTEGRATION)
    cout << "#derivatives: end." << endl;

  return GSL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argn, char *args[]){

  //initialize emulator
  if(C.use_Clpp_mt4()) Pm_MT4(-log(C.a_in()),1);
  if(argn>1) A_s_extern = atof(args[1]);
  if(C.use_Clpp_ee2()) Pm_EE2(-log(C.a_in()),1,A_s_extern);
  
  //initialize k grid
  for(int i=0; i<nk; i++){
    lnkArr[i] = lnkmin + dlnk*i;
    kArr[i] = exp(lnkArr[i]);

    if(i==0) ellArr[i] = 1;
    else if(i==nk-1) ellArr[i] = ellkapmax;
    else{
      double Dellkap_i = log(ellkapmax/ellArr[i-1]) / (nk-i);
      ellArr[i] = floor(1 + ellArr[i-1]*exp(Dellkap_i));
    }
    
    lnellArr[i] = log(ellArr[i]);
  }

  //initialize P/I array
  //double D_in[2], f_in[nk];//= a_in*D_in[1]/D_in[0]; //d ln(D) / d ln(a)
  double *y = new double[nU*nk+C.n_phi()*nk], Ddum[2];
  C.D_dD(C.z_in(),kArr[0],Ddum); //initialize function
  C.Plin_cb(C.z_in(),kArr[0]);

#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<nk; i++){

    //growth derivative d ln(D) / d ln(a)
    double D_in[2];
    C.D_dD(C.z_in(),kArr[i],D_in);
    double f_in = C.a_in()*D_in[1]/D_in[0]; //d ln(D) / d ln(a)

    //power spectrum
    double Pin_i = C.Plin_cb(C.z_in(),kArr[i]);
    y[i] = log(Pin_i); 
    y[nk+i] = log(Pin_i*f_in);
    y[2*nk+i] = log(Pin_i*f_in*f_in);
  }

  for(int i=nUP*nk; i<nU*nk+C.n_phi()*nk; i++) y[i] = 0.0;

  //gsl differential equation integrator
  long unsigned int N_EQ = nU*nk + C.n_phi()*nk;
  const double eps_abs = eabs_P, eps_rel = erel_P;
  const gsl_odeiv_step_type * T  = gsl_odeiv_step_rkf45;
  gsl_odeiv_step * s = gsl_odeiv_step_alloc(T, N_EQ);
  gsl_odeiv_control * c = gsl_odeiv_control_y_new(eps_abs, eps_rel);
  gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc(N_EQ);
  double mu = 0; 
  gsl_odeiv_system sys = {derivatives, dummy_jacobian, N_EQ, &mu};

  double eta =C.eta_in(), eta_fin =log(1.0/C.a_in()), deta =1e-2*(eta_fin-eta);

  cout << setprecision(PREC);
       
  int status = GSL_SUCCESS;

  //assemble new eta list combining requested outputs, z_phi_trg,
  //z_phi_min, and z_phi_max arrays
  int n_eta_list = C.n_eta() + 3*C.n_phi();
  double *eta_list = new double[n_eta_list];
  double *temp_eta_list = new double[n_eta_list];
  for(int j=0; j<C.n_eta(); j++) temp_eta_list[j] = C.etasteps(j);
  for(int j=0; j<C.n_phi(); j++)
    temp_eta_list[C.n_eta()+j] = -log((1.0+C.z_phi_trg(j))*C.a_in());
  for(int j=0; j<C.n_phi(); j++)
    temp_eta_list[C.n_eta()+C.n_phi()+j]
      = -log((1.0+C.z_phi_min(j))*C.a_in());
  for(int j=0; j<C.n_phi(); j++)
    temp_eta_list[C.n_eta()+2*C.n_phi()+j]
      = -log((1.0+C.z_phi_max(j))*C.a_in());
  sort(temp_eta_list, temp_eta_list+n_eta_list);
  int j_temp_eta_first = 0, j_temp_eta_list = 1;
  while(temp_eta_list[j_temp_eta_first]<0) j_temp_eta_first++;
  eta_list[0] = temp_eta_list[j_temp_eta_first];
  for(int j=j_temp_eta_first+1; j<n_eta_list; j++){
    eta_list[j_temp_eta_list] = temp_eta_list[j];
    j_temp_eta_list += (temp_eta_list[j] > temp_eta_list[j-1]);
  }
  n_eta_list = j_temp_eta_list;
  delete [] temp_eta_list;
  
  int i_eta_output = 0;
  for(int i_eta=0; i_eta<n_eta_list; i_eta++){
    double eta_next = eta_list[i_eta];

    //cout << "### main: integrating from eta=" << eta << " to "
    //	 << eta_next << endl;
    
    while((eta_next-eta)*deta > 0){
      status = gsl_odeiv_evolve_apply(e, c, s, &sys, 
				      &eta, eta_next, &deta, y);
      if(status != GSL_SUCCESS) break;
    }
    
    if(status != GSL_SUCCESS) 
      cout << "#WARNING: integrator failed, status = " << status << endl;
    
    //output power spectra
    if(eta >= C.etasteps(i_eta_output)){
      i_eta_output++;
      double a_ain=exp(eta), aeta=a_ain*C.a_in(), a2_ain2=a_ain*a_ain, 
	a3_ain3=a2_ain2*a_ain, a4_ain4=a2_ain2*a2_ain2, D_eta[2];
      double sV2_eta = C.sigmaV2(1.0/aeta-1);
      double H_eta = C.H_H0(aeta) * H0h, z_eta = 1.0/aeta-1.0;
      double H_a_10_13 = C.H_H0(10.0/13.0) * H0h;
      //cout << "### main: output at eta=" << eta
      //     << ", a=" << aeta << ", z=" << 1.0/aeta-1 << ", H=" << H_eta  
      //     << ", sigma_v^2=" << sV2_eta << endl;
      
      //use linear power spectrum at this redshift for 1-loop outputs
      double Aacdbef[64*nk], Rlabc[nUQ*nk], PTjm[9][nk], PMRn[8][nk],
	y_lin[3*nk], PT2[nk], PT4[nk], PT6[nk], PT8[nk];
      if(C.SWITCH_NONLINEAR() && C.SWITCH_1LOOP()){
	if(PRINTBIAS)
	  compute_Aacdbef_Rlabc_PTjm_PMRn_full(eta,y,Aacdbef,Rlabc,PTjm,PMRn);
	else
	  compute_Aacdbef_Rlabc_PTj_full(eta,y,Aacdbef,Rlabc,PT2,PT4,PT6,PT8);
      }
      
      //used for printing halofit outputs
      double lnPhf[nk], PHLF[nk], PMT4[nk], PEE2[nk], PTRG[nk];
      if(PRINTPMAT){
	if(C.use_Clpp_hft()) AU_halofit(aeta, lnPhf);
	for(int i=0; i<nk; i++){
          PHLF[i]=0; PMT4[i]=0; PEE2[i]=0;
          if(C.use_Clpp_mt4()) PMT4[i] = Pm_MT4(eta,kArr[i]);
	  if(C.use_Clpp_ee2()) PEE2[i] = Pm_EE2(eta,kArr[i],A_s_extern);
	  
	  double sqrtPnu = sqrt(C.Plin_nu(1.0/aeta-1,kArr[i])),
	    sqrtPcbTRG = exp(eta) * exp(0.5*y[i]),
	    sqrtPmTRG = C.f_cb()*sqrtPcbTRG + C.f_nu()*sqrtPnu;
	  PTRG[i] = sqrtPmTRG*sqrtPmTRG;

          if(C.use_Clpp_hft()){
	    double sqrtPcbHLF = exp(0.5*lnPhf[i]),
	      sqrtPmHLF =  C.f_cb()*sqrtPcbHLF + C.f_nu()*sqrtPnu;
	    PHLF[i] = sqrtPmHLF*sqrtPmHLF;
          }
	}
      }
      
      for(int i=0; i<nk; i++){
	
	//compute k-dep growth and Beta_p
	double D[2];
	C.D_dD(1.0/aeta-1,kArr[i],D);
	double aLeft=aeta*0.999, aRight=min(1.0,aeta*1.001);
	double B_eta=C.Beta_P(aeta,kArr[i]);
	double B1=C.Beta_P(1,kArr[i]), B_left=C.Beta_P(aLeft,kArr[i]), 
	  B_right=C.Beta_P(aRight,kArr[i]),
	  dlnB_dlna = (C.f_nu() < 1e-10 ? 0 
		       : (aeta/B_eta)*(B_right-B_left)/(aRight-aLeft));
	
	cout << setprecision(PREC);
	//	   << setw(WIDTH) << kArr[i];
	
	if(C.PRINTLIN())
	  cout << setw(WIDTH) << D[0]
	       << setw(WIDTH) << aeta*D[1]/D[0]
	       << setw(WIDTH) << C.Plin_cb(1.0/aeta-1,kArr[i])
	       << setw(WIDTH) << B_eta / (B1 + 1e-100)
	       << setw(WIDTH) << dlnB_dlna
	       << setw(WIDTH) << C.Plin_nu(1.0/aeta-1,kArr[i]);
	
	if(C.PRINTRSD())
	  cout << setw(WIDTH) << exp(y[i])*a2_ain2
	       << setw(WIDTH) << exp(y[nk+i])*a2_ain2
	       << setw(WIDTH) << exp(y[2*nk+i])*a2_ain2;
	
	//output the 14 unique components of each A_1loop and A
	if(PRINTA){
	  for(int iA=0; iA<14; iA++)
	    cout << setw(WIDTH) <<  Aacdbef[nAI(aU[iA],cU[iA],dU[iA], 
						bU[iA],eU[iA],fU[iA])*nk + i];
	}
	
	//output the 14 unique components of I
	if(PRINTI){
	  for(int iI=0; iI<nUI; iI++)
	    cout << setw(WIDTH) << y[(nUP+iI)*nk + i];
	}
	
	//output the P_{B,j}(k) and P_{T,j} TNS correction terms; 
	//  j is in {2,4,6} for P_{B,j} and {2,4,6,8} for P_{T,j}
	if(C.PRINTRSD() && PRINTBIAS){
	  cout << setw(WIDTH) << Pbisj(i,2,2,y)*a3_ain3
	       << setw(WIDTH) << Pbisj(i,2,1,y)*a3_ain3
	       << setw(WIDTH) << Pbisj(i,4,1,y)*a3_ain3
	       << setw(WIDTH) << Pbisj(i,4,0,y)*a3_ain3
	       << setw(WIDTH) << Pbisj(i,6,0,y)*a3_ain3
	       << setw(WIDTH) << PTjm[0][i]*a4_ain4 //j=2, m=2
	       << setw(WIDTH) << PTjm[1][i]*a4_ain4 //j=2, m=1
	       << setw(WIDTH) << PTjm[2][i]*a4_ain4 //j=2, m=0
	       << setw(WIDTH) << PTjm[3][i]*a4_ain4 //j=4, m=2
	       << setw(WIDTH) << PTjm[4][i]*a4_ain4 //j=4, m=1
	       << setw(WIDTH) << PTjm[5][i]*a4_ain4 //j=4, m=0
	       << setw(WIDTH) << PTjm[6][i]*a4_ain4 //j=6, m=1
	       << setw(WIDTH) << PTjm[7][i]*a4_ain4 //j=6, m=0
	       << setw(WIDTH) << PTjm[8][i]*a4_ain4 //j=8, m=0
	       << setw(WIDTH) << PMRn[0][i]*a4_ain4  //d^2,d
	       << setw(WIDTH) << PMRn[1][i]*a4_ain4  //d^2,t
	       << setw(WIDTH) << PMRn[2][i]*a4_ain4  //s^2,d
	       << setw(WIDTH) << PMRn[3][i]*a4_ain4  //s^2,t
	       << setw(WIDTH) << PMRn[4][i]*a4_ain4  //d^2,d^2
	       << setw(WIDTH) << PMRn[5][i]*a4_ain4  //d^2,s^2
	       << setw(WIDTH) << PMRn[6][i]*a4_ain4  //s^2,s^2
	       << setw(WIDTH) << PMRn[7][i]*a4_ain4; //3nl
	}
	if(C.PRINTRSD() && !PRINTBIAS){
	  cout << setw(WIDTH) << (Pbisj(i,2,2,y)+Pbisj(i,2,1,y))*a3_ain3
	       << setw(WIDTH) << (Pbisj(i,4,1,y)+Pbisj(i,4,0,y))*a3_ain3
	       << setw(WIDTH) << Pbisj(i,6,0,y)*a3_ain3
	       << setw(WIDTH) << PT2[i]*a4_ain4
	       << setw(WIDTH) << PT4[i]*a4_ain4
	       << setw(WIDTH) << PT6[i]*a4_ain4
	       << setw(WIDTH) << PT8[i]*a4_ain4;
	}
	
	//output integrated bispectrum Qmnabc
	if(PRINTQ){
	  for(int iB=0; iB<nUQ; iB++)
	    cout << setw(WIDTH) << y[(nUP+nUI+iB)*nk+i]*a3_ain3;
	}
	
	//output lensing potential power spectrum
	if(PRINTCLPP && i<nk){
	  cout << setw(WIDTH) << ellArr[i];
	  for(int j=0; j<C.n_phi(); j++)
	    cout << setw(WIDTH) << y[nU*nk+j*nk+i];
	}
	
	//output halofit, emulators, pt
	if(PRINTPMAT){
	  cout << setw(WIDTH) << kArr[i]
	       << setw(WIDTH) << aeta
               << setw(WIDTH) << C.Plin(z_eta,kArr[i])
               << setw(WIDTH) << PTRG[i]
               << setw(WIDTH) << PHLF[i] 
               << setw(WIDTH) << PMT4[i] 
	       << setw(WIDTH) << PEE2[i]
	    ;
	  for(int j=0; j<C.n_phi(); j++){
            double ph[] = {C.par_phi_p(j), C.par_phi_q(j),
			   C.par_phi_r(j), H_eta/H_a_10_13};
            //double Mhalo = pow(10.0,hyd_log10Mhat(C.zsteps(i_eta),kArr[i]));
            //double akino = 0.01*exp(ph[0]) * pow(Mhalo/1e14,ph[1]-1) 
            //               * pow(ph[3],ph[2]);
	    cout
            //   << setw(WIDTH) << Mhalo 
            //   << setw(WIDTH) << akino 
                 << setw(WIDTH) 
                 << hyd_sup(C.zsteps(i_eta),kArr[i],
                            C.SWITCH_BARYON_FEEDBACK(j),ph);
          }

	  if(PRINTHEATMAP){
	    double Pm_phi = 0;
	    int z_gt_ztrg = (C.zsteps(i_eta) > C.z_phi_trg(0));
	    switch(C.SWITCH_PHI_MAT(0)){
	    case 0: Pm_phi = C.Plin(C.zsteps(i_eta),kArr[i]);   break;
	    case 1: Pm_phi = PTRG[i];                           break;
	    case 2: Pm_phi = (z_gt_ztrg ? PTRG[i] : PHLF[i]);   break;
	    case 3: Pm_phi = (z_gt_ztrg ? PTRG[i] : PMT4[i]);   break;
	    default: Pm_phi = C.Plin(C.zsteps(i_eta),kArr[i]);  break;
	    }

	    double pre_phi = 9.0 * C.Omega_m()*C.Omega_m() * H0h*H0h*H0h 
	      / (4.0 * aeta*aeta*aeta * C.H_H0(aeta));
	    double eta_star = -log((1.0+z_star)*C.a_in());
	    double H0chi_star = C.H0chi(eta_star), H0chi = C.H0chi(eta);
	    double L_j[7] = {10, 30, 101, 302, 998, 2994, 10000}, g_chi_j[7];
	    for(int j=0; j<7; j++){
              double L4 = L_j[j]*L_j[j]*L_j[j]*L_j[j];
	      g_chi_j[j] = fmax(0, 2.0*(1.0 - L_j[j]*H0h/kArr[i]/H0chi_star));
	      cout << setw(WIDTH) << pre_phi/L4 *g_chi_j[j]*g_chi_j[j] *Pm_phi;
	    }
          }
	}
	  
	cout << endl;

      }//end for i

      //cout << endl;
      cout << endl << endl;

    }//end if eta

  }//end for i_eta
    
  delete [] y;
  delete [] eta_list;
  
  return 0;
}
