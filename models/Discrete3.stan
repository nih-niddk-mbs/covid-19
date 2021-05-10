// SICRdiscrete.stan
// Discrete SICR model

#include dataDiscrete.stan

transformed data {
  int segalpha  = 26;
  int segbeta  = 26;
  int segsigc  = 26;
  int segsigd  = 26;
  int segsigr  = 26;
  int n_blocksalpha = (n_weeks-1)/segalpha + 1;
  int n_blocksbeta = (n_weeks-1)/segbeta + 1;
  int n_blockssigc = (n_weeks-1)/segsigc + 1;
  int n_blockssigd = (n_weeks-1)/segsigd + 1;
  int n_blockssigr = (n_weeks-1)/segsigr + 1;
}

parameters {
real<lower=0> alpha[n_blocksalpha];           // migration rate
real<lower=0> beta[n_blocksbeta];             // infection rate
real<lower=0> sigc[n_blockssigc];              // case rate
real<lower=0> sigd[n_blockssigd];              // death rate
real<lower=0> sigr[n_blockssigr];              // recovery rate
real<lower=0> sigmau;                      // uninfected rate
real<lower=0,upper=1> ft;                          // noncase death fraction
real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
}

#include transformedparametersDiscrete.stan
#include modelDiscrete.stan
#include generatedquantitiesDiscrete.stan
