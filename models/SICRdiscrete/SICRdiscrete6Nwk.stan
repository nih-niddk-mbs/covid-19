// SICRdiscrete.stan
// Discrete SICR model

#include dataDiscrete.stan

parameters {

real<lower=0,upper=50> alpha;
real<lower=0,upper=10> beta[n_weeks];             // infection rate
real<lower=0,upper=5> sigd[n_weeks];
real<lower=0,upper=5> sigc[n_weeks];
real<lower=0,upper=5> sigr;
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
    sigmac[i] = sigc[i];
    sigmar[i] = sigr;
    sigmad[i] = sigd[i];
    I0 = s*Nt;
    I += alpha;
    I *= exp(beta[i]*s - sigmac[i] - sigmau);
    s *= exp(-beta[i]*I/Nt);
    dC[i] = sigmac[i]*I;
    C *= exp(-(sigmar[i]+sigmad[i]));
    C += dC[i];

    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;

    Ccum += dC[i];
    Dcum += dD[i];

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
    sigd ~ exponential(2.);
    sigr ~ exponential(2.);
    alpha ~ exponential(10.);
    beta ~ normal(1.,.5);
    sigc ~ exponential(1.);
    ft ~ beta(5,1);
    extra_std ~ exponential(1.);

#include likelihoodDiscrete.stan

    target += normal_lpdf(beta[2]-beta[1] | 0, .1);
    target += normal_lpdf(sigc[2]-sigc[1] | 0, .1);
    target += normal_lpdf(sigd[2]-sigd[1] | 0, .1);

    for (i in 2:n_weeks-1){
      target += normal_lpdf(beta[i+1]-beta[i] | 0, .1);
      target += normal_lpdf(sigc[i+1]-sigc[i] | 0, .1);
      target += normal_lpdf(sigd[i+1]-sigd[i] | 0, .1);
    }

    for (i in 1:n_weeks){
      target += normal_lpdf(car[i] | .1, .2);
      target += normal_lpdf(ifr[i] | .01, .01);
      target += normal_lpdf(Rt[i] | 1.5, 1.5);
      }
}

// generated quantities
#include generatedquantitiesDiscrete.stan
