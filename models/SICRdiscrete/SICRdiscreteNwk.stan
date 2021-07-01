// SICRdiscrete.stan
// Discrete SICR model

#include dataDiscrete.stan

parameters {
real<lower=0> beta[n_weeks];             // infection rate
real<lower=0> alpha[n_weeks];
real<lower=0> sigd[n_weeks];
real<lower=0> sigc[n_weeks];
real<lower=0> sigr[n_weeks];
real<lower=0> sigmau;             // uninfected rate
real<lower=0,upper=1> ft;
real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
}

transformed parameters {

  real S[n_weeks];
  real dI[n_weeks];
  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];
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
  int delay  = 2;

  s = 1;

  C = 0;
  I = 0;
  for (i in 1:n_weeks){
    sigmac[i] = sigc[i];
    sigmar[i] = sigr[i];
    sigmad[i] = sigd[i];
    I0 = s*N;
    I += alpha[i];
    I *= exp(beta[i]*s - sigmac[i] - sigmau);
    s *= exp(-beta[i]*I/N);
    dC[i] = sigmac[i]*I;
    C *= exp(-(sigmar[i]+sigmad[i]));
    C += dC[i];
    Cd[i] = C;
    dI[i] = I0 - s*N;
    S[i] = s*N;
    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;
    /*
    if (i > delay){
      dR[i] = sigmar[i]*Cd[i-delay];
      dD[i] = sigmad[i]*Cd[i-delay];
      }
    else {
      dD[i] = sigmad[i]*C;
      dR[i] = sigmar[i]*C;
      }
      */
  }
  }
}

model {

    sigmau ~ exponential(1.);

    for (i in 1:n_weeks){
      alpha[i] ~ exponential(10.);
      beta[i] ~ normal(1.,.5);
      sigd[i] ~ exponential(5.);
      sigc[i] ~ exponential(1.);
      sigr[i] ~ exponential(2.);

      #include likelihoodDiscrete.stan


    target += normal_lpdf(beta[2]-beta[1] | 0, .1);
    target += normal_lpdf(alpha[2]-alpha[1] | 0, .1);
    target += normal_lpdf(sigc[2]-sigc[1] | 0, .1);
    target += normal_lpdf(sigd[2]-sigd[1] | 0, .1);
    target += normal_lpdf(sigr[2]-sigr[1] | 0, .1);

    for (i in 2:n_weeks-1){
      target += normal_lpdf(beta[i+1]-beta[i] | 0, .1);
      target += normal_lpdf(beta[i+1]-2*beta[i]+beta[i-1] | 0, .05);
      target += normal_lpdf(alpha[i+1]-alpha[i] | 0, .1);
      target += normal_lpdf(alpha[i+1]-2*alpha[i]+alpha[i-1] | 0, .05);
      target += normal_lpdf(sigc[i+1]-sigc[i] | 0, .1);
      target += normal_lpdf(sigc[i+1]-2*sigc[i]+sigc[i-1] | 0, .05);
      target += normal_lpdf(sigd[i+1]-sigd[i] | 0, .1);
      target += normal_lpdf(sigd[i+1]-2*sigd[i]+sigd[i-1] | 0, .05);
      target += normal_lpdf(sigr[i+1]-sigr[i] | 0, .1);
      target += normal_lpdf(sigr[i+1]-2*sigr[i]+sigr[i-1] | 0, .05);

    }

}

// generated quantities
#include generatedquantitiesDiscrete.stan
