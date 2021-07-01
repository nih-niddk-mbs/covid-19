// SICRdiscreteFT.stan
// Discrete SICR model

functions {

  real ift(real[] coeff,int i,int n_weeks) {

        real param;
        int T = (size(coeff) - 1) / 2;

        for (n in 1:T) {
          param = coeff[2*n-1]*sin(2*pi()/n_weeks*n*i) + coeff[2*n]*cos(2*pi()/n_weeks*n*i);
          }
        param += coeff[1];

        return exp(param);
  }

}


data {
      int<lower=1> N;                   //population
      int<lower=1> n_obs;
      int<lower=0> n_weeks;      // number of weeks
      int n_total;               // total number of weeks simulated, n_total-n_weekss is weeks beyond last data point
      int<lower=1> n_ostates;   // number of observed states
      int y[n_weeks,n_ostates];     // data, per-week-tally [cases,recovered,death]
      real tm;                    // start day of mitigation
      real ts[n_total];             // time points that were observed + projected
  }

transformed data {

  int n_coeffs = 9;

}

parameters {
real betap[n_coeffs];
real alphap[n_coeffs];
real sigd[n_coeffs];
real sigc[n_coeffs];
real sigr[n_coeffs];
real<lower=0,upper=5> sigmau;
}

transformed parameters {

  real S[n_weeks];
  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];

  real ir[n_weeks];
  real ar[n_weeks];
  real car[n_weeks];
  real ifr[n_weeks];
  real Rt[n_weeks];

  real alpha[n_weeks];
  real beta[n_weeks];
  real sigmad[n_weeks];
  real sigmar[n_weeks];
  real sigmac[n_weeks];
{
  real C;
  real I;
  real I0;
  real s;
  real Cd[n_weeks];
  real dI;
  real Z;
  real Ccum;
  real Dcum;

  int Nt;
  int delay  = 2;

  s = 1;

  Ccum = 0;
  Dcum = 0;
  C = 0;
  I = 0;
  Z = 1;

  Nt = N;

  for (i in 1:n_weeks){
    alpha[i] = ift(alphap,i,n_weeks);
    beta[i] = ift(betap,i,n_weeks);
    sigmac[i] = ift(sigc,i,n_weeks);
    sigmar[i] = ift(sigr,i,n_weeks);
    sigmad[i] = ift(sigd,i,n_weeks);

    I0 = s*Nt;
    I += alpha[i];
    I *= exp(beta[i]*s - sigmac[i] - sigmau);
    s *= exp(-beta[i]*I/Nt);
    dC[i] = sigmac[i]*I;
    C *= exp(-(sigmar[i]+sigmad[i]));
    C += dC[i];

    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;

    Ccum += dC[i];
    Dcum += dD[i];


    /*
    Cd[i] = C;
    if (i > delay){
      dR[i] = sigmar[i]*Cd[i-delay];
      dD[i] = sigmad[i]*Cd[i-delay];
      }
    else {
      dD[i] = sigmad[i]*C;
      dR[i] = sigmar[i]*C;
      }
    */
    S[i] = s*Nt;
    Z += I0 - s*Nt;
    ir[i] = 1-s;
    ar[i] = N*ir[i]/S[i];
    car[i] = Ccum/Z;
    ifr[i] = Dcum/Z;
    Rt[i] = beta[i]*s/(sigmac[i]+sigmau);

  }
  }
}

model {

    sigmau ~ exponential(1.);
    alphap ~ normal(0.,4.);;
    betap ~  normal(0.,4.);
    sigd ~  normal(0.,4.);;
    sigc ~  normal(0.,4.);;
    sigr ~  normal(0.,4.);;

    for (i in 1:n_weeks){
      if (y[i,1] > -1)
        target += poisson_lpmf(y[i,1] | dC[i]);
      if (y[i,2] > -1)
        target += poisson_lpmf(y[i,2] | dR[i]);
      if (y[i,3] > -1)
        target += poisson_lpmf(y[i,3] | dD[i]);
    }

    for (i in 1:n_weeks){
      target += normal_lpdf(car[i] | .1, .1);
      target += normal_lpdf(ifr[i] | .01, .005);
      target += normal_lpdf(Rt[i] | 1., 1.);
      }
}

// generated quantities

generated quantities {
    int y_proj[n_weeks,n_ostates];
    real llx[n_weeks,n_ostates];
    real ll_ = 0; // log-likelihood for model

    real R0 = beta[1]/(sigc[1]+sigmau);

    {

      for (i in 1:n_weeks) {
        y_proj[i,1] = poisson_rng(min([dC[i],1e8]));
        y_proj[i,2] = poisson_rng(min([dR[i],1e8]));
        y_proj[i,3] = poisson_rng(min([dD[i],1e8]));

        if (y[i,1] > -1)
          llx[i,1] = poisson_lpmf(y[i,1] | min([dC[i]/7,1e8]));
        else
          llx[i,1] = 0.;

        if (y[i,2] > -1)
          llx[i,2] = poisson_lpmf(y[i,2] | min([dR[i]/7,1e8]));
        else
          llx[i,2] = 0.;

        if (y[i,3] > -1)
          llx[i,3] = poisson_lpmf(y[i,3] | min([dD[i]/7,1e8]));
        else
          llx[i,3] = 0.;
        ll_ += llx[i,1] + llx[i,2] + llx[i,3];
      }
    }

}
