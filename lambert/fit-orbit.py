import numpy as np
import spiceypy
import os
import io
import sys
import tqdm
import matplotlib.pyplot as plt

from astropy.time import Time

from grss import fit, prop, utils

os.chdir(os.path.join(os.getcwd(), 'code/Final-Project/'))

np.set_printoptions(precision=10, linewidth=np.inf)

au2km   = 1.495978707e8
day2sec = 86400.0

spiceypy.furnsh("../../data/naif0012.tls")
spiceypy.furnsh("../../data/de440.bsp")
# add observations by the ESA's Gaia satellite (requires special treatment)
add_gaia_obs = False

# restrict timespan observations are considered for
t_min_tdb = None
t_max_tdb = None
# use lower resolution asteroid catalog debiasing scheme (JPL default, see Eggl+ 2020)
debias_lowres = False
# deweight outlier observations rather than eliminate them
deweight = True
eliminate = False
# limit the number of observations per night used in the fit. Avoids biases toward single nights that have many observations
num_obs_per_night = 4
# verbose filtering feedback
verbose = True


# # get optical observations from file
obs = fit.get_optical_obs('SYNTH', "../../data/2024pdc25_epoch2.xml", t_min_tdb, t_max_tdb, debias_lowres, deweight, eliminate, num_obs_per_night, verbose)
obs = obs[obs['stn'] == '500']

# initial guess from IOD
position = [-4.847007e-01, -8.848794e-01, -2.827967e-01]
velocity = [ 1.599048e-02, -1.148257e-02, -1.246919e-03]
epoch    = 60465.99902532168

init_sol = {
    't': epoch, 
    'x': position[0], 
    'y': position[1],
    'z': position[2],
    'vx': velocity[0], 
    'vy': velocity[1], 
    'vz': velocity[2]
}

init_cov = 0.1*np.eye(6)

nongrav_info = {
    'a1': 0.0,
    'a2': 0.0,
    'a3': 0.0,
    'alpha': 1.0,
    'k': 0.0,
    'm': 2.0,
    'n': 0.0,
    'r0_au': 1.0,
    'radius': 0.0,
}

# activate non-gravitaional forces
add_a1 = True
add_a2 = True
add_a3 = True
ng_guess = 1e-13
if add_a1:
    # add a1 to init_sol
    init_sol['a1'] = ng_guess
    # add row and column for a1 to init_cov
    init_cov = np.pad(init_cov, ((0,1),(0,1)), mode='constant', constant_values=ng_guess**2)
if add_a2:
    # add a2 to init_sol
    init_sol['a2'] = ng_guess
    # add row and column for a2 to init_cov
    init_cov = np.pad(init_cov, ((0,1),(0,1)), mode='constant', constant_values=ng_guess**2)
if add_a3:
    # add a3 to init_sol
    init_sol['a3'] = ng_guess
    # add row and column for a3 to init_cov
    init_cov = np.pad(init_cov, ((0,1),(0,1)), mode='constant', constant_values=ng_guess**2)

# set up the fit
fit_sim = fit.FitSimulation(init_sol, obs, nongrav_info=nongrav_info, cov_init=init_cov, n_iter_max=10, de_kernel=440)
fit_sim.reject_outliers   = True
fit_sim.analytic_partials = True
fit_sim.fit_cartesian     = True
fit_sim.fit_cometary      = False

fit_sim.reject_criteria = [2.0, 1.8]

fit_sim.filter_lsq()

fit_sim.iters[1].plot_iteration_summary(title='Prefit Residuals', auto_close=True)
fit_sim.iters[-1].plot_iteration_summary(title='Postfit Residuals', auto_close=True)

mean_state = np.array((
    fit_sim.x_nom['x'],
    fit_sim.x_nom['y'],
    fit_sim.x_nom['z'],
    fit_sim.x_nom['vx'],
    fit_sim.x_nom['vy'],
    fit_sim.x_nom['vz'],
))

# we know the general date, so let's overshoot by a little bit
# April 24, 2041 => 17 years should cover it
tf = epoch + 365*18

pos0 = mean_state[0:3]
vel0 = mean_state[3:6]

# to be consistent with our earlier definition
ng_params = prop.NongravParameters()
ng_params.a1 = fit_sim.x_nom['a1']
ng_params.a2 = fit_sim.x_nom['a2']
ng_params.a3 = fit_sim.x_nom['a3']
ng_params.alpha = 1.0
ng_params.k = 0.0
ng_params.m = 2.0
ng_params.n = 0.0
ng_params.r0_au = 0.0

pdc25 = prop.IntegBody("SYNTH", epoch, 0.0, 0.0, pos0, vel0, ng_params)

de_kernel = 440
de_kernel_path = utils.default_kernel_path
prop_sim = prop.PropSimulation("PDC25 propagation", epoch, de_kernel, de_kernel_path)

# t_eval = np.linspace(epoch, tf, int(tf - epoch))
# t_eval = np.arange(epoch, tf, 1.0)
t_eval = []

eval_apparent_state = False
t_eval_utc = False
converged_light_time = False
prop_sim.set_integration_parameters(tf, t_eval, t_eval_utc, eval_apparent_state, converged_light_time)
prop_sim.add_integ_body(pdc25)
prop_sim.integrate()

# prop_sim.interpolate(t_eval[0])
with open('../../data/pdc2025_traj.csv', 'w') as f:
    f.write('mjd_tdb,x,y,z,vx,vy,vz\n')
    for t in np.arange(epoch, tf, 1.0):
        state = prop_sim.interpolate(t)
        print(state)
        f.write(f'{t:.6f},{state[0]:.12f},{state[1]:.12f},{state[2]:.12f},{state[3]:.12f},{state[4]:.12f},{state[5]:.12f}\n')