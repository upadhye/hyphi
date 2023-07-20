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

const int AU_SWITCH_HF_BNU = 1;
const int AU_SWITCH_HF_LNU = 1;
const int AU_SWITCH_HF_NNU = 1;

const int AU_SWITCH_HF_DEBUG = 0;

double sigmaG2_minus_1(const double *D2_lnksig){
  double R = exp(-D2_lnksig[np]);
  double integrand_sigmaG2[np];
  for(int jk=0; jk<np; jk++){
    double k = exp(lnk_pad_min+dlnk*jk), x = R*k, WG = exp(-0.5*x*x);
    integrand_sigmaG2[jk] = D2_lnksig[jk] * WG*WG;
  }
  double sigmaG2 = ncint_cf(np, dlnk, integrand_sigmaG2);
  return sigmaG2 - 1.0;
}

int AU_halofit(double a, double *lnPhf){
  
  //find non-linear scale; if it's larger than threshhold, output Plin
  const	double ln_k_sig_thresh = log(100.0);
  double z = 1.0/a-1.0, D2_lnksig[np+1];
  for(int i=0; i<np; i++){
    double lnk = lnk_pad_min + dlnk*i, k = exp(lnk), pre = k*k*k/(2.0*M_PI*M_PI);
    D2_lnksig[i] = pre * C.Plin_cb(z,k);
  }
  D2_lnksig[np] = ln_k_sig_thresh;
  if(sigmaG2_minus_1(D2_lnksig)<0){
    for(int i=0; i<nk; i++) lnPhf[i] = log(C.Plin_cb(z,kArr[i]));
    return 1;
  }
  double ksig = exp( AU_bisection(&sigmaG2_minus_1, np+1, np, D2_lnksig,
				  log(1e-3), ln_k_sig_thresh, 1e-4) );

  //effective spectral index and curvature
  double sig2p_int[np], sig2pp_int[np];
  for(int i=0; i<np; i++){
    double k = exp(lnk_pad_min+dlnk*i), Rsig = 1.0/ksig, x = k*Rsig;
    double WG = exp(-0.5*x*x), WGp = -x*WG, WGpp = (x*x-1.0)*WG;
    sig2p_int[i] = D2_lnksig[i] * 2.0*k * WG * WGp;
    sig2pp_int[i] = D2_lnksig[i] * 2.0*k*k * (WGp*WGp + WG*WGpp);
  }
  double sig2p=ncint_cf(np,dlnk,sig2p_int), sig2pp=ncint_cf(np,dlnk,sig2pp_int);
  double nsig = -sig2p/ksig - 3.0, nsig2 = nsig*nsig, nsig3 = nsig2*nsig;
  double Csig = -sig2p/ksig + (sig2p*sig2p-sig2pp)/(ksig*ksig);

  if(AU_SWITCH_HF_DEBUG)
    cout << "#AU_halofit: a=" << a << ", ksig=" << ksig << ", nsig=" << nsig
         << ", Csig=" << Csig << endl;

  //standard Bird(2012) halofit without fnu-dependent corrections
  double Om_a = C.Omega_m() / (a*a*a * C.H2_H02(a));
  double f1_HF = pow(Om_a, -0.0307);
  double f2_HF = pow(Om_a, -0.0585);
  double f3_HF = pow(Om_a, 0.0743);
  
  double alpha_HF = 1.38848 + 0.3701*nsig - 0.1452*nsig2,
    beta_HF = 0.8291 + 0.9854*nsig + 0.3400*nsig2,
    gamma_HF = 1.18075 + 0.2224*nsig - 0.6719*Csig,  
    a_HF = pow(10.0, 1.4861 + 1.83693*nsig + 1.67618*nsig2 + 0.7940*nsig3 
               + 0.1670756*nsig2*nsig2 - 0.620695*Csig),
    b_HF = pow(10.0, 0.9463 + 0.9466*nsig + 0.3084*nsig2 - 0.940*Csig),
    c_HF = pow(10.0, -0.2807 + 0.6669*nsig + 0.3214*nsig2 - 0.0793*Csig),
    mu_HF = pow(10.0, -3.54419 + 0.19086*nsig), 
    nu_HF = pow(10.0, 0.95897 + 1.2857*nsig);

  //fnu-dependent corrections
  double Nnu_HF = 0.977 * AU_SWITCH_HF_NNU,
    Bnu_HF = (-6.4868 + 1.4373*nsig2) * AU_SWITCH_HF_BNU;
  beta_HF += C.f_nu() * Bnu_HF;

  if(AU_SWITCH_HF_DEBUG)
    cout << "#AU_halofit: a_HF=" << a_HF
	 << ", b_HF=" << b_HF
	 << ", c_HF=" << c_HF
	 << ", alpha_HF=" << alpha_HF
	 << ", beta_HF=" << beta_HF
	 << ", gamma_HF=" << gamma_HF
	 << ", mu_HF=" << mu_HF
	 << ", nu_HF=" << nu_HF
	 << endl;

  //halofit
  for(int i=0; i<nk; i++){
    double k = kArr[i], y = k/ksig, f_HF = 0.25*y+0.125*y*y;
    double Lnu_HF = 47.48*k*k*C.h()*C.h() * AU_SWITCH_HF_LNU
      / (1.0 + 1.5*k*k*C.h()*C.h());
    double D2_Q_HF = D2_lnksig[i+nshift] * exp(-f_HF)
      * pow(1.0 + D2_lnksig[i+nshift]*(1.0+C.f_nu()*Lnu_HF), beta_HF)
      / (1.0 + alpha_HF * D2_lnksig[i+nshift] * (1.0 + C.f_nu()*Lnu_HF));
    double D2_Hp_HF = a_HF * pow(y,3.0*f1_HF)
      / (1.0 + b_HF*pow(y,f2_HF) + pow(c_HF*f3_HF*y, 3.0-gamma_HF));
    double D2_H_HF = D2_Hp_HF * (1.0 + C.f_nu()*Nnu_HF)
      / (1.0 + mu_HF/y + nu_HF/(y*y));
    double D2cb_HF = D2_Q_HF + D2_H_HF;
    lnPhf[i] = log( D2cb_HF * 2.0*M_PI*M_PI / (k*k*k) );
  }

  return 0;
}
