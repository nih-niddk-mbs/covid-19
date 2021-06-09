// SICRdiscrete.stan
// Discrete SICR model

#include dataDiscrete.stan

transformed data {
  int seg  = 1;
  int n_blocks = (n_weeks-1)/seg + 1;
}

parameters {
real<lower=0,upper=200> alpha;
real<lower=0,upper=10> beta[n_blocks];             // infection rate
real<lower=0,upper=5> sigd[n_blocks];
real<lower=0,upper=5> sigc[n_blocks];
real<lower=0,upper=5> sigr[n_blocks];
real<lower=0,upper=5> sigmau;             // uninfected rate
real<lower=0,upper=1> ft;
real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
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
  real phi = max([1/(extra_std^2),1e-10]); // likelihood over-dispersion of std
{
  real sigmad[n_weeks];
  real sigmar[n_weeks];
  real sigmac[n_weeks];
  real beta_wk[n_weeks];
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
  int k;
  int delay  = 2;

  s = 1;

  Ccum = 0;
  Dcum = 0;
  C = 0;
  I = 0;
  Z = 1;

  Nt = N;

  for (i in 1:n_weeks){
    k = (i-1)/seg + 1;
    sigmac[i] = sigc[k];
    sigmar[i] = sigr[k];
    sigmad[i] = sigd[k];
    beta_wk[i] = beta[k];

    I0 = s*Nt;
    I += alpha;
    I *= exp(beta_wk[i]*s - sigmac[i] - sigmau);
    s *= exp(-beta_wk[i]*I/Nt);
    dC[i] = sigmac[i]*I;
    C += dC[i];
    C *= exp(-(sigmar[i]+sigmad[i]));

    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;

    Ccum += dC[i];
    Dcum += dD[i];

    S[i] = s*Nt;
    Z = Nt-S[i];
    ir[i] = 1-s;
    ar[i] = N*ir[i]/S[i];
    car[i] = Ccum/Z;
    ifr[i] = Dcum/Z;
    Rt[i] = beta_wk[i]*s/(sigmac[i]+sigmau);

  }
  }
}

//model
#include modelDiscrete.stan

// generated quantities
#include generatedquantitiesDiscrete.stan
