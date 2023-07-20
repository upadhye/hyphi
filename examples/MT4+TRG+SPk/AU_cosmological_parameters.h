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
#include <string>
#include <cmath>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

#include "AU_tabfun.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// For faster performance, the following parameter changes may be made:      //
//                                                                           //
//   * In Beta_P(double, input_data):  k_min = 1e-3, k_max = 1               //
//                                                                           //
//   * In D_dD(double,input_data,double*):  n_lnk = 50, a_early = 1e-20      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

//camb transfer function type: 0 is new 13-column file, 1 is old 7-column
const int SWITCH_TRANSFER_TYPE = 0;

class cosmological_parameters{

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
 private: /////////////////////////////////////////////////////////////////////

  //debug switches
  static const int DEBUG_LINEAR = 0;
  static const int DEBUG_INTERP = 0;
  static const int DEBUG_INPUT  = 0;

  //maximum number of transfer functions to read
  static const int MAXTRANSFER = 200;

  //cosmological parameters and input parameters
  string tc_c, tnu_root, *z_transfer_str;
  int init, S_NL, S_1L, S_PL, S_PR, S_NU, n_interp_z, n_eta_inp;
  int n_Clpp, *S_Clpp_Pmat, *S_Clpp_hyd_sup;
  int USE_CLPP_HFT=0, USE_CLPP_MT4=0, USE_CLPP_EE2=0;
  double ns_c, s8_c, h0_c, Om_c, Ob_c, On_c, On_hot_c, TK_c, w0_c, wa_c;
  double Og_c, fnu_c, fcb_c, anu_c, Or_c, OL_c, a_in_inp, z_in_inp;
  double *asteps_inp, *zsteps_inp, *etasteps_inp, *z_transfer_flt;
  double *z_Clpp_trg, z_Clpp_trg_min=1e9, *z_Clpp_min, *z_Clpp_max;
  double *par_Clpp_p, *par_Clpp_q, *par_Clpp_r;
  static constexpr double C_rho_gam = 4.46911743913795e-07;
  static constexpr double C_nu_hot = 0.681321952980717;  //3.0*(7.0/8.0) * pow(4.0/\11.0,4.0/3.0);

  //computation of comoving distance vs. eta = ln(a/a_in)
  static const int N_H0CHI = 1000;
  double eta_chi_i[N_H0CHI], H0chi_i[N_H0CHI];
  
  //column conventions for standard CAMB transfer function files.  nVars
  //is the total number of columns, while i_k, i_dc, i_db, and i_dnu are
  //the columns corresponding to k, delta_c, delta_b, and delta_nu, in
  //C-style notation (so that the first column number is zero).
  static const int nVars = (SWITCH_TRANSFER_TYPE==0 ? 13 : 7);
  static const int i_k = 0, i_dc = 1, i_db = 2, i_dnu = 5;

  //some useful functions
  static inline void discard_comments(std::ifstream *file){
    while(file->peek()=='#' || file->peek()=='\n'){file->ignore(10000,'\n');} }
 
  static double fdiff(double x, double y){
    return 2.0*fabs(x-y)/(fabs(x)+fabs(y)); }

  //structure to hold all input data; used for static functions
  struct input_data{
    int n_z;
    double p[11];
    //double H0chi_vs_eta[N_H0CHI];
    string TcbFile;
    string TnuRoot;
    string zT[MAXTRANSFER];
  };

  static int input_data_copy(const input_data from, input_data *to){
    to->n_z = from.n_z;
    to->TcbFile = from.TcbFile;
    to->TnuRoot = from.TnuRoot;
    for(int i=0; i<11; i++) to->p[i] = from.p[i];
    //for(int i=0; i<N_H0CHI; i++) to->H0chi_vs_eta[i] = from.H0chi_vs_eta[i];
    for(int i=0; i<from.n_z; i++) to->zT[i] = from.zT[i];
    return 0;
  }

  int input_data_init(input_data *dat0){
    dat0->p[0] = n_s();
    dat0->p[1] = sigma_8();
    dat0->p[2] = h();
    dat0->p[3] = Omega_m();
    dat0->p[4] = Omega_b();
    dat0->p[5] = Omega_nu();
    dat0->p[6] = Omega_L();
    dat0->p[7] = T_cmb_K();
    dat0->p[8] = w0();
    dat0->p[9] = wa();
    dat0->p[10]= 0.1; //pick something
    //for(int i=0; i<N_H0CHI; i++) H0chi_vs_eta[i] = 0;
    dat0->n_z = n_interp_z;
    dat0->TcbFile = tc_c;
    dat0->TnuRoot = tnu_root;
    for(int j=0; j<n_interp_z; j++) dat0->zT[j] = z_transfer_str[j];
    return 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  static int func_to_integrate_d_dDda_kdep(double a, const double y[], 
					   double f[], void *params){

    //get params from vector
    input_data *dat = (input_data *)params;
    double p[11]; 
    for(int j=0; j<11; j++) p[j] = dat->p[j];
    double k0 = p[10], fn = p[5] / p[3], fc = 1.0 - fn;

    //General Relativity factors (for CDM)
    double F_GR_0 = 1.5 * p[3] / (pow(a,5) * H2_H02(a,p));
    double F_GR_1 = (3.0 + dlnH_dlna(a,p)) / a;

    //neutrino contribution Beta_P = f_nu delta_nu / delta_cdm
    double Beta = (a<1e-3 ? fn : Beta_P(fmin(a,1),*dat));

    //modified gravity contribution
    const double beta_mg = 0, m_mg_a = 0;
    double F_MG = 2.0 * k0*k0 * beta_mg*beta_mg / (k0*k0 + a*a*m_mg_a*m_mg_a);

    //growth factor and derivative
    f[0] = y[1];
    f[1] = -F_GR_1*y[1] + F_GR_0 * (fc + Beta) * (1.0 + F_MG) * y[0];

    return GSL_SUCCESS;
  }

  static int dummy_jacobian(double t, const double y[], double *dfdy, 
			    double dfdt[], void *params){ return GSL_SUCCESS; }

  static int integrate_growth(double a_begin, double a_end, 
			      input_data *d, double *y){
    const gsl_odeiv_step_type * T  = gsl_odeiv_step_rk8pd;
    double err_abs = 0, err_rel = 1e-6;
    gsl_odeiv_step * s = gsl_odeiv_step_alloc(T, 2);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new(err_abs, err_rel);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc(2);
    gsl_odeiv_system sys = {func_to_integrate_d_dDda_kdep,
			    dummy_jacobian, 2, d};
    double t = a_begin, t1 = a_end, dt = 1e-6*t;
    int status = 0;
    while((t1-t)*dt > 0){
      status = gsl_odeiv_evolve_apply(e, c, s, &sys, &t, t1, &dt, y);
      if(status != GSL_SUCCESS) break;
    }
    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);
    return 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  //comoving distance
  static double H0chi_integrand(double z, void *params){
    double *p = (double *)params;
    double aeta = 1.0 / (1.0 + z), H_H0_eta = H_H0(aeta, p);
    return 1.0 / H_H0_eta;
  }

  /////////////////////////////////////////////////////////////////////////////
  //linear power spectrum

  //total power spectrum (unnormalized), used to find sigma_8
  static double Plin_integrand_unnorm(double lnkR, void *params){
    //un-norm linear power spectrum times Window function ^2 * k^2/(2pi^2)
    input_data *dat = (input_data *)params;
    const double R = 8.0;
    double kR = exp(lnkR), kR2 = kR*kR, kR3 = kR2*kR, k = kR/R;
    dat->p[10] = k;
    double T = Transfer_cb(*dat), F = 1.0-dat->p[5]/dat->p[3] + Beta_P(1,*dat);
    double W = 1.0 - 0.1*kR*kR;
    if(kR > 1e-2) W = 3.0 * (sin(kR)/kR3 - cos(kR)/kR2);
    return W*W * T*T * F*F * pow(k,dat->p[0]+3.0) / (2.0*M_PI*M_PI);
  }

  static double sigmaV2_integrand(double lnk, void *params){ 
    input_data *dat = (input_data *)params;
    dat->p[10] = exp(lnk);
    return exp(lnk)*Plin(0,*dat); 
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
 public:///////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  ////////// INITIALIZATION

  cosmological_parameters(string paramfile_inp){
    const int n_f_par = 9, n_i_par = 4; //numbers of float/int params
    int temp_i[n_i_par];
    double temp_f[n_f_par];
    
    if(DEBUG_INPUT)
      std::cout << "#cosmological_parameters: opening parameter file: "
      	        << paramfile_inp.c_str() << endl;

    std::ifstream input(paramfile_inp.c_str());

    //read cosmological params
    for(int i=0; i<n_f_par; i++){
      cosmological_parameters::discard_comments(&input);
      input >> temp_f[i];
      if(DEBUG_INPUT) 
	std::cout << "#cosmological_parameters: read parameter " << i
                  << ": " << temp_f[i] << std::endl;
    }

    //read switches
    for(int i=0; i<n_i_par; i++){
      cosmological_parameters::discard_comments(&input);
      input >> temp_i[i];
      if(DEBUG_INPUT) 
        std::cout << "#cosmological_parameters: read switch " << i 
                  << ": " << temp_i[i] << std::endl;
    }

    //read initial redshift
    cosmological_parameters::discard_comments(&input);
    input >> z_in_inp;
    a_in_inp = 1.0 / (1.0 + z_in_inp);
    if(DEBUG_INPUT)
        std::cout << "#cosmological_parameters: read z_in=" 
	          << z_in_inp << std::endl;

    //read output redshifts: number and list
    cosmological_parameters::discard_comments(&input);
    input >> n_eta_inp;
    cosmological_parameters::discard_comments(&input);
    asteps_inp = new double[n_eta_inp];
    zsteps_inp = new double[n_eta_inp];
    etasteps_inp = new double[n_eta_inp];
    for(int i=0; i<n_eta_inp; i++){
      input >> zsteps_inp[i];
      asteps_inp[i] = 1.0 / (1.0+zsteps_inp[i]);
      etasteps_inp[i] = log(asteps_inp[i]/a_in_inp);
      if(DEBUG_INPUT)
        std::cout << "#cosmological_parameters: z_" << i << " = "
		  << zsteps_inp[i] << std::endl; 
    } 

    //read transfer inputs
    cosmological_parameters::discard_comments(&input);
    input >> tc_c;
    if(DEBUG_INPUT)
      std::cout << "#cosmological_parameters: transfer file: "
		<< tc_c.c_str() << std::endl; 

    int neut_interp_type = -100;
    cosmological_parameters::discard_comments(&input);
    input >> neut_interp_type;
    if(neut_interp_type != 0) abort(); //ONLY DO THIS CASE SO FAR

    cosmological_parameters::discard_comments(&input);
    input >> tnu_root;

    n_interp_z = -100;
    cosmological_parameters::discard_comments(&input);
    input >> n_interp_z;
    if(n_interp_z < 0) abort();
    if(n_interp_z > MAXTRANSFER){
      std::cout << "ERROR: Number of transfer function inputs ("
		<< n_interp_z << ") exceeds MAXTRANSFER=" << MAXTRANSFER
		<< ".  Increase MAXTRANSFER and recompile." << std::endl;
      abort();
    }

    z_transfer_str = new string[n_interp_z];
    z_transfer_flt = new double[n_interp_z];
    cosmological_parameters::discard_comments(&input);
    for(int i=0; i<n_interp_z; i++){ 
      input >> z_transfer_str[i];
      z_transfer_flt[i] = atof(z_transfer_str[i].c_str());
      if(DEBUG_INPUT)
        std::cout << "#cosmological_parameters: nu transfer redshift " << i
		  << ": " << z_transfer_flt[i] << std::endl;
    }

    //read lensing potential power and baryon feedback inputs
    cosmological_parameters::discard_comments(&input);
    input >> n_Clpp;
    S_Clpp_Pmat = new int[n_Clpp];
    S_Clpp_hyd_sup = new int[n_Clpp];
    z_Clpp_trg = new double[n_Clpp];
    z_Clpp_min = new double[n_Clpp];
    z_Clpp_max = new double[n_Clpp];
    par_Clpp_p = new double[n_Clpp];
    par_Clpp_q = new double[n_Clpp];
    par_Clpp_r = new double[n_Clpp];
 
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++){ 
      input >> S_Clpp_Pmat[j];
      if(S_Clpp_Pmat[j]==1) z_Clpp_trg_min=0;
      if(S_Clpp_Pmat[j]==2) USE_CLPP_HFT=1;    
      if(S_Clpp_Pmat[j]==3) USE_CLPP_MT4=1;
      if(S_Clpp_Pmat[j]==4) USE_CLPP_EE2=1;
    }

    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> S_Clpp_hyd_sup[j];
      
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++){ 
      input >> z_Clpp_trg[j];
      if(z_Clpp_trg[j] < z_Clpp_trg_min) z_Clpp_trg_min=z_Clpp_trg[j];
    } 

    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> z_Clpp_min[j];
        
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> z_Clpp_max[j];
        
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> par_Clpp_p[j];
    
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> par_Clpp_q[j];
 
    cosmological_parameters::discard_comments(&input);
    for(int j=0; j<n_Clpp; j++) input >> par_Clpp_r[j];
 
    input.close();

    if(DEBUG_INPUT){
      std::cout << "#cosmological_parameters: n_Clpp="
		<< n_Clpp << " lensing potential power spectra:" << std::endl;
      for(int j=0; j<n_Clpp; j++)
	std::cout << "#cosmological_parameters: j=" << j
		  << ", lowk=" << S_Clpp_Pmat[j]
		  << ", hyd_sup=" << S_Clpp_hyd_sup[j]
		  << ", ztrg=" << z_Clpp_trg[j]
		  << ", zmin=" << z_Clpp_min[j]
		  << ", zmax=" << z_Clpp_max[j]
		  << ", hyd_pf=" << par_Clpp_p[j] 
                  << ", hyd_qf=" << par_Clpp_q[j] 
                  << ", hyd_rf=" << par_Clpp_r[j] 
                  << std::endl;
    }
    
    //input parameters
    ns_c = temp_f[0];
    s8_c = temp_f[1];
    h0_c = temp_f[2];
    Om_c = temp_f[3];
    Ob_c = temp_f[4];
    On_c = temp_f[5];
    TK_c = temp_f[6];
    w0_c = temp_f[7];
    wa_c = temp_f[8];
    
    //code switches
    S_NL = temp_i[0];//SWITCH_NL;
    S_1L = temp_i[1];//SWITCH_1L;
    S_PL = temp_i[2];//SWITCH_LIN;
    S_PR = temp_i[3];//SWITCH_RSD;
    S_NU = neut_interp_type;
    
    //derived parameters
    Og_c = C_rho_gam * (TK_c*TK_c*TK_c*TK_c) / (h0_c*h0_c);
    fnu_c = On_c / Om_c;
    fcb_c = 1.0 - fnu_c;
    On_hot_c = C_nu_hot * Og_c;
    anu_c = C_nu_hot * Og_c / (fnu_c*Om_c + 1e-15); //nu cold for a>=a_nu
    Or_c = Og_c + On_hot_c*(anu_c>1.0);
    OL_c = 1.0 - Om_c - Or_c;
    
    //initialized!  Don't change parameters after this.
    init = 1;
  }  

  /////////////////////////////////////////////////////////////////////////////
  /////// COSMOLOGICAL PARAMETERS

  //parameter-output functions
  double n_s(){ return ns_c; }
  double sigma_8(){ return s8_c; }
  double h(){ return h0_c; }
  double Omega_m(){ return Om_c; }
  double Omega_b(){ return Ob_c; }
  double Omega_nu(){ return On_c; }
  double Omega_L(){ return OL_c; }
  double T_cmb_K(){ return TK_c; }
  double w0(){ return w0_c; }
  double wa(){ return wa_c; }
  double f_nu(){ return fnu_c; }
  double f_cb(){ return fcb_c; }

  //CB transfer function file
  string transferfile(){ return tc_c; }
  int transferfile(char *tfile){ strcpy(tfile,tc_c.c_str()); return 0; }

  //initial condition and time stepping information
  double z_in(){ return z_in_inp; }
  double a_in(){ return a_in_inp; }
  double eta_in(){ return 0; } //by definition
  int n_eta(){ return n_eta_inp; }
  double zsteps(int i){ return zsteps_inp[i]; }
  double asteps(int i){ return asteps_inp[i]; }
  double etasteps(int i){ return etasteps_inp[i]; }

  //lensing potential power spectrum parameters
  int n_phi(){ return n_Clpp; }
  int use_Clpp_hft(){ return USE_CLPP_HFT; }
  int use_Clpp_mt4(){ return USE_CLPP_MT4; }
  int use_Clpp_ee2(){ return USE_CLPP_EE2; }
  double z_phi_trg(int j){ return z_Clpp_trg[j]; }
  double z_phi_trg_min() { return z_Clpp_trg_min;}
  double z_phi_min(int j){ return z_Clpp_min[j]; }
  double z_phi_max(int j){ return z_Clpp_max[j]; }
  double par_phi_p(int j){ return par_Clpp_p[j]; }  
  double par_phi_q(int j){ return par_Clpp_q[j]; }
  double par_phi_r(int j){ return par_Clpp_r[j]; }

  //cosmic evolution functions, and static copies (for GSL)
  //  The vector of cosmological parameters, used in static functions, is
  //      i = 0  1 2  3  4  5  6  7  8  9 10
  //    p[i]=ns s8 h Om Ob On OL Tc w0 wa  k
  double w(double a){ return w0_c + wa_c*(1.0-a); } //DE eos
  static double w(double a, double *p){ return p[8] + p[9]*(1.0-a); }

  double E(double a){ //E=rho_DE(a)/rho_DE(1)
    return std::pow(a,-3.0*(1.0+w0_c+wa_c)) * exp(-3.0*wa_c*(1.0-a)); }
  static double E(double a, double *p){
    return std::pow(a,-3.0*(1.0+p[8]+p[9])) * exp(-3.0*p[9]*(1.0-a)); }
  
  double dEda(double a){ 
    return 3.0 * E(a) * (wa_c - (1.0+w0_c+wa_c) / a); }
  static double dEda(double a, double *p){
    return 3.0 * E(a,p) * (p[9] - (1.0+p[8]+p[9]) / a); }

  double dE_dlna(double a){ return a*dEda(a); }
  static double dE_dlna(double a, double *p){ return a*dEda(a,p); }

  double d2E_dlna2(double a){
    return 3.0*a*dEda(a)*(wa_c*a - (1.0+w0_c+wa_c)) + 3.0*E(a)*wa_c*a; }
  static double d2E_dlna2(double a, double *p){
    return 3.0*a*dEda(a,p)*(p[9]*a - (1.0+p[8]+p[9])) + 3.0*E(a,p)*p[9]*a; }
  
  double Y(double a){ //Y=rho_nu(a)/rho_cb(a)
    if(a>=anu_c) return fnu_c / fcb_c; //cold
    return C_nu_hot * Og_c / (fcb_c * Om_c * a); //hot 
  }
  static double Y(double a, double *p){
    double Og = C_rho_gam * std::pow(p[7]*p[7]/p[2],2);
    double fn = p[5] / p[3], fc = 1.0 - fn;
    double anu = C_nu_hot * Og / (p[5]+1e-15); 
    if(a>=anu) return fn / fc; //cold
    return C_nu_hot * Og / (fc * p[3] * a); //hot
  }
  
  double dYda(double a){
    if(a>=anu_c) return 0;
    return -C_nu_hot * Og_c / (fcb_c * Om_c * a*a);
  }
  static double dYda(double a, double *p){
    double Og = C_rho_gam * std::pow(p[7]*p[7]/p[2],2);
    double anu = C_nu_hot * Og / (p[5]+1e-15);
    if(a>=anu) return 0;
    return -C_nu_hot * Og / ( (p[3]-p[5]) * a*a);
  }

  double H2_H02(double a){
    return fcb_c*Om_c*(1.0+Y(a))/pow(a,3) + OL_c*E(a) + Og_c/pow(a,4); }
  static double H2_H02(double a, double *p){
    double Og = C_rho_gam * std::pow(p[7]*p[7]/p[2],2);
    return (p[3]-p[5])*(1.0+Y(a,p))/pow(a,3) + p[6]*E(a,p) + Og/pow(a,4); 
  }
  
  double H_H0(double a){ return sqrt(H2_H02(a)); }
  static double H_H0(double a, double *p){ return sqrt(H2_H02(a,p)); }

  double dlnH_dlna(double a){
    return 0.5*a/H2_H02(a) 
      * ( fcb_c*Om_c * (-3.0*(1.0+Y(a))+a*dYda(a)) / std::pow(a,4)
	  + OL_c*dEda(a) 
	  - 4.0*Og_c/std::pow(a,5) );
  }
  static double dlnH_dlna(double a, double *p){
    double Og = C_rho_gam * std::pow(p[7]*p[7]/p[2],2);
    double fn = p[5] / p[3], fc = 1.0 - fn;
    return 0.5*a/H2_H02(a,p)
      * ( fc*p[3] * (-3.0*(1.0+Y(a,p))+a*dYda(a,p)) / std::pow(a,4)
          + p[6]*dEda(a,p)
          - 4.0*Og/std::pow(a,5) );
  }

  //time-dependent Omega_m
  double Omega_m(double a){ return Om_c / (a*a*a * H2_H02(a)); }
  static double Omega_m(double a, double *p){ 
    return p[3] / (a*a*a * H2_H02(a,p)); }

  //code switches
  int SWITCH_NONLINEAR(){ return S_NL; }
  int SWITCH_1LOOP(){ return S_1L; }
  int PRINTLIN(){ return S_PL; }
  int PRINTRSD(){ return S_PR; }
  int NEUT_INTERP(){ return S_NU; }
  int SWITCH_PHI_MAT(int j){ return S_Clpp_Pmat[j]; }
  int SWITCH_BARYON_FEEDBACK(int j){ return S_Clpp_hyd_sup[j]; }

  /////////////////////////////////////////////////////////////////////////////
  ////////// GROWTH FACTOR CALCULATION WITH NEUTRINO INTERP

  //Beta(a,k) as defined in Pietroni 2008
  static double Beta_P(double a, input_data dat){

    //get parameters from p
    double fn = dat.p[5] / dat.p[3], fc = 1.0 - fn;
    double k = dat.p[10];

    //if not using CAMB interpolation, return 0
    if(dat.n_z == 0) return 0;

    //if no neutrinos, Beta_P = 0
    if(fn < 1e-10) return 0;

    //if a is a little bit greater than 1, call with a=1 (CAMB doesn't do a>1)
    if(a > 1.001){
      std::cout << "ERROR in Beta_P: a>1." << std::endl;
      abort();
    }
    if(a > 1.0) return Beta_P(1, dat);

    //Beta shouldn't change with k for k>> k_max and k<<k_min
    //const double k_min = 1e-5, k_max = 20.0;
    const double k_min = 1e-3, k_max = 1.0;
    if(k<k_min){ dat.p[10]=k_min; return Beta_P(a,dat); }
    if(k>k_max){ dat.p[10]=k_max; return Beta_P(a,dat); }

    //initialize
    const int n_k0 = 30000;
    static int init = 0;
    static tabulated_function BetaInterp;
    if(!init){
      double kArr[n_k0], BetaArr0[n_k0], temp[nVars];
      double *aArr = new double[dat.n_z];

      //i=0, and read n_k
      aArr[0] = 1.0 / (1.0 + atof(dat.zT[0].c_str()));
      string filename0 = dat.TnuRoot + dat.zT[0] + ".dat";
      ifstream inputTable0(filename0.c_str(), ios::in);
      cosmological_parameters::discard_comments(&inputTable0);
      int status0 = 1, j0 = 0;

      do{
	for(int jk=0; jk<nVars; jk++) 
	  status0 = status0 && (inputTable0 >> temp[jk]);
	cosmological_parameters::discard_comments(&inputTable0);

	if(status0){
	  kArr[j0] = temp[i_k];
	  BetaArr0[j0] = fn * temp[i_dnu] / temp[i_dc];

	  if(DEBUG_INTERP)
	    std::cout << "#Beta_P: "
                    << filename0
		      << setw(20) << aArr[0]
		      << setw(20) << kArr[j0]
		      << setw(20) << BetaArr0[j0]
		      << std::endl;
	}
      } while(status0 && ++j0<n_k0);

      inputTable0.close();

      //initialize Beta array
      int n_k = j0;
      double *BetaArr = new double[dat.n_z*n_k];
      for(int j=0; j<n_k; j++) BetaArr[j] = BetaArr0[j];

      //rest of the z's
      for(int i=1; i<dat.n_z; i++){ 
	aArr[i] = 1.0 / (1.0 + atof(dat.zT[i].c_str())); 
	string filename = dat.TnuRoot + dat.zT[i] + ".dat";
	ifstream inputTable(filename.c_str(), ios::in);
	cosmological_parameters::discard_comments(&inputTable);
	int status = 1, j = 0;

	do{
	  for(int jk=0; jk<nVars; jk++) 
	    status = status && (inputTable >> temp[jk]);
	  cosmological_parameters::discard_comments(&inputTable);

	  if(status){
	    if(fdiff(kArr[j],temp[i_k]) > 1e-5){
	      std::cout << "#Beta_P: ERROR: Initialization failed.  k lists "
			<< "in transfer function inputs are not the same." 
			  << endl;
	      abort();
	    }
	    
	    BetaArr[i*n_k + j] = fn * temp[i_dnu] / temp[i_dc];
	    
	    if(DEBUG_INTERP)
	      std::cout << "#Beta_P: "
			<< filename
			<< setw(20) << aArr[i]
			<< setw(20) << kArr[j]
			<< setw(20) << BetaArr[i*n_k + j]
			<< std::endl;
	  }
	} while(status && ++j<n_k);
	
	inputTable.close();
      }
      
      BetaInterp.initialize(dat.n_z,n_k,aArr,kArr,BetaArr);
      delete [] aArr;
      delete [] BetaArr;
      init = 1;
    }

    return BetaInterp.f(a,k);
  }

  double Beta_P(double a, double k){
    input_data d;
    input_data_init(&d);
    d.p[10] = k;
    return Beta_P(a,d);
  }

  static int D_dD(double z, input_data d, double *D_dDda){
  
    //check bounds
    double k = d.p[10];
    const double a_min=1e-3, a_max=1.1;
    double a_int = 1.0/(z+1.0);
    if(a_int > a_max || a_int < a_min){
      cout << "D_dD: ERROR: z=" << z << " out of bounds." << endl;
      abort();
    }

    const double k_min=1.5e-4, k_max = 9.0;
    if(k>k_max){ d.p[10]=k_max; return D_dD(z, d, D_dDda); }
    if(k<k_min){ d.p[10]=k_min; return D_dD(z, d, D_dDda); }

    //initialize
    static int init = 0;
    //const int n_lna = 100, n_lnk = 1000, n_tot = (n_lna+1)*(n_lnk+1);
    const int n_lna = 100, n_lnk = 50, n_tot = (n_lna+1)*(n_lnk+1);
    static tabulated_function Dnorm(n_lnk+1); //normalization
    static tabulated_function G_lna_lnk(n_lna+1,n_lnk+1); //G(ln a, ln k) = D/a
    static tabulated_function dDda_lna_lnk(n_lna+1,n_lnk+1); //dD/da(ln a,ln k)

    if(!init){

      //force Beta_P interpolation before parallelizing
      Beta_P(0.1,d);

      //a and k arrays
      double lna_min=log(a_min),lna_max=log(a_max),dlna=log(a_max/a_min)/n_lna;
      double lnk_min=log(k_min),lnk_max=log(k_max),dlnk=log(k_max/k_min)/n_lnk;
      double lnaTab[n_lna+1], lnkTab[n_lnk+1];
      double *GTab = new double[n_tot];
      double *dDdaTab = new double[n_tot]; 
      for(int i=0; i<=n_lna; i++) lnaTab[i] = lna_min + dlna*i;
      for(int j=0; j<=n_lnk; j++) lnkTab[j] = lnk_min + dlnk*j;

      //loop over k's first so we can integrate in time
#pragma omp parallel for schedule(dynamic)
      for(int j=0; j<=n_lnk; j++){
	//double params[] = {exp(lnkTab[j])};
	input_data dat;
	input_data_copy(d,&dat);
	dat.p[10] = exp(lnkTab[j]);
	//double a_early = 1e-50;
        double a_early = 1e-20;
	double y[] = {1.0, 1.0/a_early}; //D and dD/da

	//i=0 step
	integrate_growth(a_early,a_min,&dat,y);
	GTab[j] = y[0] / a_min;
	dDdaTab[j] = y[1];

	for(int i=1; i<=n_lna; i++){
	  integrate_growth(exp(lnaTab[i-1]),exp(lnaTab[i]),&dat,y);
	  GTab[i*(n_lnk+1)+j] = y[0] / exp(lnaTab[i]);
	  dDdaTab[i*(n_lnk+1)+j] = y[1];
	}
      }

      //fill interpolation tables, and get normalization
      G_lna_lnk.input_arrays(lnaTab,lnkTab,GTab);
      dDda_lna_lnk.input_arrays(lnaTab,lnkTab,dDdaTab);
      double DnormTab[n_lnk+1];
      for(int j=0; j<=n_lnk; j++) DnormTab[j] = G_lna_lnk(0,lnkTab[j]); 
      Dnorm.input_arrays(lnkTab,DnormTab);

      //clean up, and update init
      delete [] GTab;
      delete [] dDdaTab;
      init = 1;
    }

    //now that we've initialized, get D and dDda
    double lna0 = log(a_int), lnk0 = log(k), D0 = Dnorm(lnk0);
    D_dDda[0] = G_lna_lnk(lna0,lnk0) * a_int / D0;
    D_dDda[1] = dDda_lna_lnk(lna0,lnk0) / D0;
    return 0;
  }

  int D_dD(double z, double k, double *D_dDda){
    input_data d;
    input_data_init(&d);
    d.p[10] = k;
    return D_dD(z,d,D_dDda);
  }

  /////////////////////////////////////////////////////////////////////////////
  //comoving distance
  int H0chi_eta_init(void){

    double zmin=1e-4, zmax=1e4, dlnz = log(zmax/zmin) / (N_H0CHI-1), zlast=0;
    input_data dat;
    input_data_init(&dat);
    
    for(int i=0; i<N_H0CHI; i++){
      double z = zmin * exp(dlnz * i), aeta = 1.0/(1.0+z), DH0chi, dum;
      eta_chi_i[N_H0CHI-1-i] = log(aeta / a_in_inp);

      gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
      gsl_function F;
      F.function = &H0chi_integrand;
      F.params = &dat.p;
      gsl_integration_qag(&F, zlast, z, 0, 1e-4, 1000, 6, w, &DH0chi, &dum);

      zlast = z;
      H0chi_i[N_H0CHI-1-i] = (i==0 ? DH0chi : DH0chi + H0chi_i[N_H0CHI-i]);
    }

    return 0;
  }

  double H0chi(double eta){
    double aeta = a_in_inp * exp(eta), zaeta = 1.0/aeta - 1.0;
    if(zaeta <= 1e-4) return zaeta;
    static int init = 0;
    if(!init){ H0chi_eta_init(); init=1; }
    static tabulated_function H0chiInterp(N_H0CHI, eta_chi_i, H0chi_i);
    return H0chiInterp(eta);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  ////////// LINEAR TRANSFER FUNCTION AND POWER SPECTRUM

  //linear power spectrum (CDM+baryon); at first call, read transfer fn. file
  static double Transfer_cb(const input_data dat){

    if(DEBUG_LINEAR)
    cout << "#Transfer begin.  Called with k=" << dat.p[10] << endl;

    //initialize
    static int init = 0;
    static tabulated_function log_transfer;

    if(!init){

      //read transfer function
      int nTk = 0, status = 1;
      double kTi[100000], lnkTi[100000], Ti[100000], lnTi[100000], temp[nVars];
      double f_b_cb = dat.p[4] / (dat.p[3]-dat.p[5]), f_c_cb = 1.0-f_b_cb;
      ifstream transfer_input(dat.TcbFile.c_str());

      discard_comments(&transfer_input);
      for(int i=0; i<nVars; i++) status = status && transfer_input >> temp[i];
      while(status){
	discard_comments(&transfer_input);
	kTi[nTk] = temp[i_k];
	lnkTi[nTk] = log(kTi[nTk]);
	double T_c = temp[i_dc], T_b = temp[i_db];
	Ti[nTk] = f_b_cb*T_b + f_c_cb*T_c;
	lnTi[nTk] = log(Ti[nTk]/Ti[0]);
        nTk++;
	for(int i=0; i<nVars; i++) status = status && transfer_input >>temp[i];
      }
      transfer_input.close();

      log_transfer.initialize(nTk, lnkTi, lnTi);
      init = 1;
    }

    if(DEBUG_LINEAR) 
    cout << "#Transfer end. returning T=" 
	 << exp(log_transfer(log(dat.p[10]))) << endl;

    return exp(log_transfer(log(dat.p[10])));
  }

  static double Plin(double z, input_data dat0){
  
    if(DEBUG_LINEAR)
      cout << "#Plin begin. Called with z=" << z 
	   << " and k=" << dat0.p[10] << endl;

    //initialize: normalize to given sigma_8
    static int init = 0;
    static double Norm = 0;
    if(!init){

      int ws_size = 1000;
      int int_type = 6;
      double x0=-15, x1=15;
      double epsabs = 0, epsrel = 1e-4;
      double result, error;
      double dummy = 0;
      input_data dat;
      input_data_copy(dat0, &dat);

      gsl_integration_workspace *w = gsl_integration_workspace_alloc(ws_size);
   
      gsl_function F;
      F.function = &Plin_integrand_unnorm;
      F.params = &dat;
  
      gsl_integration_qag(&F,             //function 
			  x0,  x1,        //interval
			  epsabs, epsrel, //error bounds
			  ws_size,        //size of integration workspace
			  int_type,       //type of numerical integration used
			  w, 
			  &result, 
			  &error); 
  
      gsl_integration_workspace_free(w);

      Norm = dat.p[1] * dat.p[1] / result;
      init = 1;

      if( DEBUG_LINEAR)
	cout << "#Plin: found Norm=" << Norm << endl;
    }

    double T=Transfer_cb(dat0), DdD[]={0,0}; 
    double F = 1.0-dat0.p[5]/dat0.p[3] + Beta_P(1.0/(1.0+z),dat0);
    D_dD(z,dat0,DdD);

    if( DEBUG_LINEAR)
    cout << "#Plin end. Returning Plin=" 
         << Norm * pow(dat0.p[10],dat0.p[0]) * T*T * DdD[0]*DdD[0] << endl;

    return Norm * pow(dat0.p[10],dat0.p[0]) * T*T * F*F * DdD[0]*DdD[0];
  }

  double Plin(double z, double k){
    input_data d;
    input_data_init(&d);
    d.p[10] = k;
    return Plin(z,d);
  }

  double Plin_nu(double z, input_data dat){
    double fn = dat.p[5]/dat.p[3], fc=1.0-fn;
    if(fn <= 1e-10) return 0;
    double a=1.0/(1.0+z), B=Beta_P(a,dat), F=fc+B, R=B/fn/F; 
    return Plin(z,dat) * R*R;
  }

  double Plin_nu(double z, double k){
    input_data d;
    input_data_init(&d);
    d.p[10] = k;
    return Plin_nu(z,d);
  }

  double Plin_cb(double z, input_data dat){
    double fn = dat.p[5]/dat.p[3], fc=1.0-fn;
    if(fn <= 1e-10) return Plin(z,dat);
    double a=1.0/(1.0+z), R = 1.0 / (fc + Beta_P(a,dat));
    return Plin(z,dat) * R*R;
  }

  double Plin_cb(double z, double k){
    input_data d;
    input_data_init(&d);
    d.p[10] = k;
    return Plin_cb(z,d);
  }

  double sigmaV2(double z){

    if( DEBUG_LINEAR)
      cout << "#sigmaV2: begin.  called with z=" << z << endl;

    static int init=0;
    static double sigmaV2_z0=0;
    if(!init){
      int ws_size = 1000;
      int int_type = 6;
      double x0=-15, x1=15;
      double epsabs = 0, epsrel = 1e-4;
      double result, error;
      double sixPi2=6.0*M_PI*M_PI;

      //store input data in form accessible to static functions
      input_data dat;
      input_data_init(&dat);

      gsl_integration_workspace *w = gsl_integration_workspace_alloc(ws_size);

      gsl_function F;
      F.function = &sigmaV2_integrand;
      F.params = &dat;

      gsl_integration_qag(&F,
			  x0,  x1, 
			  epsabs, epsrel,
			  ws_size,
			  int_type,
			  w,
			  &result,
			  &error);

      gsl_integration_workspace_free(w);
      sigmaV2_z0 = result / sixPi2;
      init = 1;
    }
    double D[2], kmin_sv2=1e-3;
    D_dD(z,kmin_sv2,D);

    if(DEBUG_LINEAR)
      cout << "#sigmaV2: end.  returning " << D[0]*D[0]*sigmaV2_z0 << endl;

    return D[0]*D[0]*sigmaV2_z0;
  }

  //clean up
  ~cosmological_parameters(){
    delete [] z_transfer_str;
    delete [] z_transfer_flt;
    delete [] asteps_inp;
    delete [] zsteps_inp;
    delete [] etasteps_inp;
    delete [] S_Clpp_Pmat;
    delete [] S_Clpp_hyd_sup;
    delete [] z_Clpp_trg;
    delete [] z_Clpp_min;
    delete [] z_Clpp_max;
    delete [] par_Clpp_p;
    delete [] par_Clpp_q;
    delete [] par_Clpp_r;
  }
};

