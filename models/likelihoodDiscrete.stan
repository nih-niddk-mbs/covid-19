
    for (i in 1:n_weeks){
      if (y[i,1] > -1)
        target += neg_binomial_2_lpmf(y[i,1]| dC[i],phi);
      if (y[i,2] > -1)
        target += neg_binomial_2_lpmf(y[i,2]| dR[i],phi);
      if (y[i,3] > -1)
        target += neg_binomial_2_lpmf(y[i,3]| dD[i],phi);
    }
