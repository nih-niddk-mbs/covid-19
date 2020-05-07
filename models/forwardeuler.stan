functions {
    real[ , ] integrate_ode_euler(
        real[] initial_state, real initial_time,
        real[] times, real[] theta, real[] x_r, int[] x_i, int inner_steps)
        {   
            // simple and stupid forward euler
            // assumes that ode_rhs is defined as a function with signature real[]
            /*
            
            */
            int N_T = size(times);
            int N_states = size(initial_state);
            real y[N_T, N_states];
            real t;

            real y_temp[N_states];
            real tf  = initial_time;
            real t0;
            real dt;
            real ydot[N_states];
            for (state in 1:N_states){
                y_temp[state] = initial_state[state];
            }

            for (timestep in 1:N_T){
                t0 = tf;
                tf = times[timestep];
                dt = (tf-t0)/inner_steps;
                for (inner_step in 1:inner_steps){
                    t = t0 + dt*(inner_step-1);
                    ydot = ode_rhs(t, y_temp, theta, x_r, x_i);
                    for (state in 1:N_states){
                        y_temp[state] = y_temp[state] + dt*ydot[state];                        
                    }
                }
                for (state in 1:N_states){
                    y[timestep, state] = y_temp[state];
                }
            }
            return y;
        }
}