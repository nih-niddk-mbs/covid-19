// SICRdiscrete.stan
// Discrete SICR model


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

vector[10] mu = rep_vector(0, 10);



}

parameters {

vector[10] theta;

real<lower=0> alpha;

}

transformed parameters {

  vector[10] bexp;
  matrix[10, 10] L_K;
  matrix[10, 10] K = cov_exp_quad(ts[1:10], 1, 3.);
  L_K = cholesky_decompose(K);

  bexp = L_K*theta;
  print(K);
  print(L_K);

}

model {



    theta ~ std_normal();

    alpha ~ exponential(10.);



}

// generated quantities

generated quantities {

}
