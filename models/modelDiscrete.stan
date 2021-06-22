
model {

    alpha ~ exponential(10.);
    beta ~ normal(1.,.5);
    sigmau ~ exponential(1.);
    sigd ~ exponential(5.);
    sigr ~ exponential(2.);
    sigc ~ exponential(10.);
    ft ~ beta(5,1);
    extra_std ~ exponential(1.);

    for (i in 1:n_weeks){
      if (dC[i] > 1e7 || dR[i] > 1e7 || dD[i] > 1e7 || dC[i] < 0 || dR[i] < 0 || dD[i] < 0 || is_nan(-dC[i]) || is_nan(-dR[i])|| is_nan(-dD[i]))
        target += 1000000;
      else {
        if (y[i,1] > -1)
          target += neg_binomial_2_lpmf(y[i,1]/scale| dC[i],phi);
        if (y[i,2] > -1)
          target += neg_binomial_2_lpmf(y[i,2]/scale| dR[i],phi);
        if (y[i,3] > -1)
          target += neg_binomial_2_lpmf(y[i,3]/scale| dD[i],phi);
        }
    }
    for (i in 1:n_blockssigc-1){
      target += normal_lpdf((sigc[i+1]-sigc[i])/sigc[i] | 0, .05);
      target += normal_lpdf((beta[i+1]-beta[i])/beta[i] | 0, .05);
    }
    for (i in 1:n_weeks){
      if (is_nan(car[i]) || is_nan(ifr[i]) || is_nan(Rt[i]))
        target += 1000000;
      else {
        target += normal_lpdf(car[i] | .1,.4);
        target += normal_lpdf(ifr[i] | .01, .04);
        target += normal_lpdf(Rt[i] | 1.,4.);
      //  target += normal_lpdf(car[i] | .1,.2);
      //  target += normal_lpdf(ifr[i] | .01, .02);
      //  target += normal_lpdf(Rt[i] | 1.,2.);
        }
      }
}
