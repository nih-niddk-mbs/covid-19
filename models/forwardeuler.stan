functions {
    real[ , ] integrate_ode_euler(
        real[] initial_state, real initial_time,
        real[] times, real[] theta, real[] x_r, int[] x_i, inner_steps=20)
        {   
            // simple and stupid forward euler
            // assumes that ode_rhs is defined as a function with signature real[]
            /*
            
            */
            int N_T = size(times) + 1;
            int N_states = size(initial_state);
            real y[N_T, N_states];
            real t;

            real y_temp[N_states];
            real tf  = initial_time;
            real t0;
            for (state in 1:N_states){
                // first timepoint is the intial value
                y[1, state] = initial_state[state];
                y_temp[state] = initial_state[state];
            }

            for (timestep in 2:N_T){
                real t0 = tf;
                real tf = times[timestep - 1];
                real dt = (tf-t0)/inner_steps;
                for (inner_step in 1:inner_steps){
                    t = t0 + dt*(inner_step-1);
                    real[] ydot = ode_rhs(t, y_temp, theta, x_r, x_i);
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