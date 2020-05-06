// Latent variable SIR model with q=0 (i.e. perfect quarantine of cases)
// include time dependence in sigmac to model change in case criteria or differences in testing


functions { // time transition functions for beta and sigmac
                real transition(real base,real location,real t) {
                   return base + (1-base)/(1 + exp(.2*(t - location)));
                }

            // SIR model
            real[] SIR(
            real t,             // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters
            real[] x_r,
            int[] x_i
            )
            {
            real du_dt[5];
            real f1 = theta[1];          // beta - sigmau - sigmac
            real f2 = theta[2];          // beta - sigma u
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real sigmau = theta[5];
            real q = theta[6];
            real mbase = theta[7];
            real mlocation = theta[8];
            real cbase = theta[9];
            real clocation = theta[10];

            real sigma = sigmar + sigmad;
            real sigmac = f2/(1+f1);
            real beta = f2 + sigmau;

            real I = u[1];  // infected, latent
            real C = u[2];  // cases, observed

            sigmac *= cbase + (1-cbase)/(1 + exp(.2*(t - clocation)));  // case detection change
            beta *= mbase + (1-mbase)/(1 + exp(.2*(t - mlocation)));  // mitigation

            du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - sigma*C;                 //C
            du_dt[3] = beta*(I+q*C);                       //N_I
            du_dt[4] = sigmac*I;                           // N_C case appearance rate
            du_dt[5] = C;                                  // integrated C

            return du_dt;
          }
        }

        data {
          int<lower=1> n_difeq;       // number of differential equations for yhat
          real<lower=1> n_scale;      // scale variables for numerical stability
          real tm;                    // start day of mitigation
          
          int<lower=1> n_obs;         // number of days observed (includes missing)
          real ts[n_obs];             // time points that were observed (includes missing)
        
          int<lower=1> n_obs_c;       // number of days observed - cases
          int<lower=1> n_obs_r;       // number of days observed - recovered
          int<lower=1> n_obs_d;       // number of days observed - deaths
          
          int ts_c[n_obs_c];         // time points that were observed - cases
          int ts_r[n_obs_r];                         // time points that were observed - recovered
          int ts_d[n_obs_d];         // time points that were observed - deaths
          
          int y_c[n_obs_c];             // new per day - cases
          int y_r[n_obs_r];             // new per day - recovered
          int y_d[n_obs_d];             // new per day - deaths
          
         //real rel_tol;             // relative tolerance for ODE solver
          //real max_num_steps;       // for ODE solver
          }

        transformed data {
            real x_r[0];
            int x_i[0];
            real q = 0.;
        }

        parameters {
          // model parameters
            real<lower=0> f1;       //initital infected to case ratio
            real<lower=0> f2;       // f2  beta - sigmau
            real<lower=0> sigmar;   // recovery rate
            real<lower=0> sigmad;   // death rate
            real<lower=0> sigmau;   // I disapperance rate
            //real<lower=0> q;              // infection factor for cases
            real<lower=0> mbase;          // mitigation factor
            real<lower=0> mlocation;      // day of mitigation application
            real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
            real<lower=0> cbase;          // case detection factor
            real<lower=0> clocation;      // day of case change
        }

        transformed parameters{
            real lambda[n_obs,3];
        
            real lambda_c[n_obs_c];     //neg_binomial_2 rate - cases
            real lambda_r[n_obs_r];     //neg_binomial_2 rate - recovered
            real lambda_d[n_obs_d];     //neg_binomial_2 rate - deaths
            
            real u_init[n_difeq];     // initial conditions for odes
            real sigmac = f2/(1+f1);
            real beta = f2 + sigmau;
            real sigma = sigmar + sigmad;
            real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number
            real phi = 1/(extra_std^2);  // likelihood over-dispersion of std

          {  // local
             real theta[10] = {f1,f2,sigmar,sigmad,sigmau,q,mbase,mlocation,cbase,clocation};
             real u[n_obs, n_difeq];   // solution from the ODE solver
             real sigmact;
             real betat;
             int t;

             real cinit = y_c[1]/n_scale;

             u_init[1] = f1*cinit;      // I set from f1 * C initial
             u_init[2] = cinit;         // C  from data
             u_init[3] = u_init[1];     // N_I cumulative infected
             u_init[4] = cinit;         // N_C total cumulative cases
             u_init[5] = cinit;         // integral of active C

            // integrate over all days (including putative missing)
            u = integrate_ode_rk45(SIR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-3,1e-3,2000);


            //C
            t = ts_c[1];
            lambda_c[1] = (u[t,4]-u_init[4])*n_scale; //C: cases per day
            for(i in 2:n_obs_c){
                t = ts_c[i];
                lambda_c[i] = (u[t,4]-u[t-1,4])*n_scale; //C: cases per day
            }
            
            //R
            if(n_obs_r>1)
            {
                t = ts_r[1];
                if(t==1){
                    lambda_r[1] = sigmar*(u[t,5]-u_init[5])*n_scale; //R: cases per day
                }
                else {
                    lambda_r[1] = sigmar*(u[t,5]-u[t-1,5])*n_scale; //R: recovered per day;
                }
                for(i in 2:n_obs_r){
                    t = ts_r[i];
                    lambda_r[i] = sigmar*(u[t,5]-u[t-1,5])*n_scale; //R: recovered per day
                }
            }
            
            //D
            t = ts_d[1];
            if(t==1){
                lambda_d[1] = sigmad*(u[t,5]-u_init[5])*n_scale; //D: cases per day
            }
            else {
                lambda_d[1] = sigmad*(u[t,5]-u[t-1,5])*n_scale; //D: recovered per day;
            }
            
            for(i in 2:n_obs_d){
                t = ts_d[i];
                lambda_d[i] = sigmad*(u[t,5]-u[t-1,5])*n_scale; //D: recovered per day
            }
            
            lambda[1,1] = (u[1,4]-u_init[4])*n_scale; //C: cases per day
            lambda[1,2] = sigmar*(u[1,5]-u_init[5])*n_scale; //R: recovered per day
            lambda[1,3] = sigmad*(u[1,5]-u_init[5])*n_scale; //D: dead per day

            for (i in 2:n_obs){

            lambda[i,1] = (u[i,4]-u[i-1,4])*n_scale; //C: cases per day
            lambda[i,2] = sigmar*(u[i,5]-u[i-1,5])*n_scale; //R: recovered rate per day
            lambda[i,3] = sigmad*(u[i,5]-u[i-1,5])*n_scale; //D: dead rate per day
            }
            
            }
        }

        model {
            // Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
            f1 ~ gamma(2.,1./10.);                 // f1  initital infected to case ratio
            f2 ~ gamma(1.5,1.);                    // f2  beta - sigmau
            sigmar ~ inv_gamma(4.,.2);             // sigmar
            sigmad ~ inv_gamma(2.78,.185);         // sigmad
            sigmau ~ inv_gamma(2.3,.15);           // sigmau
            q ~ exponential(5.);                   // q
            mbase ~ exponential(4.);               // mbase
            mlocation ~ lognormal(log(tm+5),1.);   // mlocation
            extra_std ~ exponential(1.);           // likelihood over dispersion std
            cbase ~ exponential(.2);               // mbase
            clocation ~ lognormal(log(20.),2.);    // mlocation

            for(i in 1:n_obs_c){
                target += neg_binomial_2_lpmf(max(y_c[i],0)|max([lambda_c[i],1.0]),phi); //C
            }
            
            if(n_obs_r>1){
                for(i in 1:n_obs_r){
                    target += neg_binomial_2_lpmf(max(y_r[i],0)|max([lambda_r[i],1.0]),phi); //R
                }
            }
            for(i in 1:n_obs_d){
                target += neg_binomial_2_lpmf(max(y_d[i],0)|max([lambda_d[i],1.0]),phi); //D
            }


        }

        generated quantities {
            }
