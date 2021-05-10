
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
      if (dC[i] > 1e8 || dR[i] > 1e8 || dD[i] > 1e8)
        target += negative_infinity();
      else {
        if (y[i,1] > -1)
          target += neg_binomial_2_lpmf(y[i,1]| dC[i],phi);
        if (y[i,2] > -1)
          target += neg_binomial_2_lpmf(y[i,2]| dR[i],phi);
        if (y[i,3] > -1)
          target += neg_binomial_2_lpmf(y[i,3]| dD[i],phi);
        }
    }

    for (i in 1:n_weeks){
      target += normal_lpdf(car[i] | .1,.2);
      target += normal_lpdf(ifr[i] | .01, .02);
      target += normal_lpdf(Rt[i] | 1.,2.);
      }
}
