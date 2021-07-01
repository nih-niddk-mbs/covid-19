// SICRdiscrete.stan
// Discrete SICR model


data {
      int<lower=1> N;                   //population
      int<lower=0> n_weeks;      // number of weeks
      int<lower=1> n_ostates;   // number of observed states
      int y[n_weeks,n_ostates];     // data, per-week-tally [cases,recovered,death]
      real ts[n_weeks];             // time points that were observed + projected
  }

transformed data {
vector[n_weeks] mu = rep_vector(0, n_weeks);
}

parameters {

real<lower=0> alpha;
real<lower=0> sigmau;             // uninfected rate
real<lower=0> bscale;
real<lower=0> cscale;
real<lower=0> rscale;
real<lower=0> dscale;

vector[n_weeks] eta;
vector[n_weeks] etac;
vector[n_weeks] etar;
vector[n_weeks] etad;

}

transformed parameters {

  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];

  real ir[n_weeks];
  real car[n_weeks];
  real ifr[n_weeks];
  real Rt[n_weeks];
  real beta[n_weeks];
  vector[n_weeks] sigmac;
  vector[n_weeks] sigmad;
  vector[n_weeks] sigmar;


{
  vector[n_weeks] cexp;
  vector[n_weeks] bexp;
  vector[n_weeks] rexp;
  vector[n_weeks] dexp;
  matrix[n_weeks, n_weeks] L_K;
  matrix[n_weeks, n_weeks] L_Kb;
  matrix[n_weeks, n_weeks] K = cov_exp_quad(ts[1:n_weeks], .1, 3.);
  matrix[n_weeks, n_weeks] Kb = cov_exp_quad(ts[1:n_weeks], .5, 3.);


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

  L_K = cholesky_decompose(K);
  L_Kb = cholesky_decompose(Kb);
  bexp = L_Kb*eta;
  cexp = L_K*etac;
  rexp = L_K*etar;
  dexp = L_K*etad;

  s = 1;
  Ccum = 0;
  Dcum = 0;
  C = 0;
  I = 0;
  Z = 1;
  Nt = N;

  for (i in 1:n_weeks){
    beta[i] =    bscale * exp(bexp[i])/(1+exp(bexp[i]));
    sigmac[i] =  cscale * exp(cexp[i])/(1+exp(cexp[i]));
    sigmar[i] =  .05 * exp(rexp[i])/(1+exp(rexp[i]));
    sigmad[i] =  .03 * exp(dexp[i])/(1+exp(dexp[i]));

    I0 = s*Nt;
    I *= exp(beta[i]*s - sigmac[i] - sigmau);
    I += alpha/(1+alpha);
    s *= exp(-beta[i]*I/Nt);
    dC[i] = sigmac[i]*I;
    C *= exp(-(sigmar[i]+sigmad[i]));
    C += dC[i];

    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;

    Ccum += dC[i];
    Dcum += dD[i];

    Z += I0 - s*Nt;
    ir[i] = 1-s;
    car[i] = Ccum/Z;
    ifr[i] = Dcum/Z;
    Rt[i] = beta[i]*s/(sigmac[i]+sigmau);

  }
  }
}

model {

    eta ~ std_normal();
    etac ~ std_normal();
    etar ~ std_normal();
    etad ~ std_normal();
    alpha ~ exponential(10.);

    bscale ~ exponential(1.);
    cscale ~ exponential(5.);
    rscale ~ exponential(10.);
    dscale ~ exponential(20.);

    sigmau ~ exponential(1.);

    for (i in 1:n_weeks){
      target += poisson_lpmf(y[i,1] | dC[i]);
      target += poisson_lpmf(y[i,2] | dR[i]);
      target += poisson_lpmf(y[i,3] | dD[i]);
    }

    for (i in 1:n_weeks){
      target += exponential_lpdf(car[i] | 1.);
      target += exponential_lpdf(ifr[i] | 1.);
      target += exponential_lpdf(Rt[i] | 1.);
      }

}

// generated quantities

generated quantities {
    int y_proj[n_weeks,n_ostates];
    real llx[n_weeks,n_ostates];
    real ll_ = 0; // log-likelihood for model
    real R0 = beta[1]/(sigmac[1]+sigmau);

    {

      for (i in 1:n_weeks) {
        y_proj[i,1] = poisson_rng(min([dC[i],1e8]));
        y_proj[i,2] = poisson_rng(min([dR[i],1e8]));
        y_proj[i,3] = poisson_rng(min([dD[i],1e8]));
        llx[i,1] = poisson_lpmf(y[i,1] | min([dC[i]/7,1e8]));
        llx[i,2] = poisson_lpmf(y[i,2] | min([dR[i]/7,1e8]));
        llx[i,3] = poisson_lpmf(y[i,3] | min([dD[i]/7,1e8]));
        ll_ += llx[i,1] + llx[i,2] + llx[i,3];
      }
    }

}
