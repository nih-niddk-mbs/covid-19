# https://blog.revolutionanalytics.com/2009/01/using-r-as-a-scripting-language-with-rscript.html
# https://mc-stan.org/rstan/articles/rstan.html

#!/usr/bin/env Rscript
library(rstan)
library(ggplot2)
library(bayesplot)

args = commandArgs(trailingOnly=TRUE)

theme_set(bayesplot::theme_default())

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


##### PYSTAN CODE TO PORT TO RSTAN
control = {'adapt_delta': args.adapt_delta}
stanrunmodel = ncs.load_or_compile_stan_model(args.model_name,
                                              args.models_path,
                                              force_recompile=args.force_recompile)

# Fit Stan
fit = stanrunmodel.sampling(data=stan_data, init=init_fun, control=control,
                            chains=args.n_chains,
                            chain_id=np.arange(args.n_chains),
                            warmup=args.n_warmups, iter=args.n_iter,
                            thin=args.n_thin)


# Uncomment to print fit summary
print(fit)

# Save fit
save_dir = Path(args.fits_path)
save_dir.mkdir(parents=True, exist_ok=True)
if args.fit_format == 0:
    save_path = save_dir / ("%s_%s.csv" % (args.model_name, args.roi))
    result = fit.to_dataframe().to_csv(save_path)
else:
    save_path = save_dir / ("%s_%s.pkl" % (args.model_name, args.roi))
    with open(save_path, "wb") as f:
        pickle.dump({'model_name': args.model_name,
                     'model_code': stanrunmodel.model_code, 'fit': fit},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

print("Finished %s" % args.roi)
