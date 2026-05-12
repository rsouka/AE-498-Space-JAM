import os
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import grss
import numpy as np
from astropy.time import Time
from grss import fit, prop, utils, libgrss


body_id = "2024PDC25"

# Initial orbit solution 

init_sol = {
    "t": 2461002.5 - 2400000.5,
    "e": 3.903977728972607E-01,
    "q": 1.005918824517983,
    "tp": 2461215.136873333567 - 2400000.5,
    "om": 214.4170705068864 * np.pi / 180,
    "w": 359.941369522211 * np.pi / 180,
    "i": 10.68653275799602 * np.pi / 180,
    "a2": 1.004247567426106E-13,
}

src = [
    -4.729401494307664E-10, 4.999699999910745E-9, -8.857690106814095E-9,
    1.768819076233289E-9, -2.647063337774079E-9, -1.094039784814247E-6,
    -1.420690625988797E-8, 2.48148634199217E-8, -1.311424078242986E-6,
    -3.353689722153545E-8, -9.430736211018382E-11, 3.390839818277143E-10,
    -3.022218031075639E-7, 1.206343342257748E-7, -1.19473371420242E-7,
    -5.364100246208384E-9, 9.054632576841123E-9, -4.548772648982771E-8,
    -2.407314566673439E-7, 2.265029335178455E-7, -8.8170973597317E-9,
    -4.887982511228615E-10, 1.345541782988126E-10, -1.065924436219504E-6,
    9.09374621143591E-10, -4.023622604787796E-9, 2.750785802423358E-11,
    3.287557722527732E-14
]


def upper_tri_src2full(vec):
    n_val = int((-1 + np.sqrt(1 - 4 * -2 * len(vec))) / 2)
    src_mat = np.zeros((n_val, n_val))

    k = 0
    for i in range(1, n_val + 1):
        for j in range(i):
            src_mat[j, i - 1] = vec[k]
            k += 1

    return src_mat @ src_mat.T


init_cov = upper_tri_src2full(src)

nongrav_info = {
    "a1": 0.0,
    "a2": init_sol["a2"],
    "a3": 0.0,
    "alpha": 1.0,
    "k": 0.0,
    "m": 2.0,
    "n": 0.0,
    "r0_au": 1.0,
    "radius": 75.0,
}

mass = 0.0


def make_nongrav_params(sol, nongrav_defaults):
    ng = prop.NongravParameters()

    ng.a1 = sol.get("a1", nongrav_defaults["a1"])
    ng.a1Est = "a1" in sol

    ng.a2 = sol.get("a2", nongrav_defaults["a2"])
    ng.a2Est = "a2" in sol

    ng.a3 = sol.get("a3", nongrav_defaults["a3"])
    ng.a3Est = "a3" in sol

    ng.alpha = nongrav_defaults["alpha"]
    ng.k = nongrav_defaults["k"]
    ng.m = nongrav_defaults["m"]
    ng.n = nongrav_defaults["n"]
    ng.r0_au = nongrav_defaults["r0_au"]

    return ng


ng_params = make_nongrav_params(init_sol, nongrav_info)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
optical_obs_file = f"./data/2024pdc25_epoch2.xml"

t_min_tdb = None
t_max_tdb = None
debias_lowres = None
deweight = False
eliminate = False
num_obs_per_night = 4
verbose = False
accept_weights = True
n_iter_max = 25

prior_est = {"a2": 1.0E-13}
prior_sig = {"a2": 3.3E-14}

# Orbit fit 

obs_df = fit.get_optical_obs(
    body_id,
    optical_obs_file,
    t_min_tdb,
    t_max_tdb,
    debias_lowres,
    deweight,
    eliminate,
    num_obs_per_night,
    verbose,
    accept_weights
)

obs_df["permID"] = body_id
obs_df["sigRA"] = obs_df["rmsRA"].values
obs_df["sigDec"] = obs_df["rmsDec"].values

obs_df.loc[obs_df["sigCorr"].isna(), "sigCorr"] = 0.0
obs_df.loc[obs_df["sigTime"].isna(), "sigTime"] = 1.0

fit_sim = fit.FitSimulation(
    init_sol,
    obs_df,
    init_cov,
    n_iter_max=n_iter_max,
    de_kernel=440,
    nongrav_info=nongrav_info,
)

fit_sim.name = body_id
fit_sim.prior_est = prior_est
fit_sim.prior_sig = prior_sig
fit_sim.analytic_partials = True
fit_sim.filter_lsq()

od_log_file = f"./{body_id}_od.log"
if os.path.exists(od_log_file):
    os.remove(od_log_file)

fit_sim.save(od_log_file)
fit_sim.print_summary()

grss_sol = {"t": fit_sim.t_sol} | fit_sim.x_nom
grss_cov = fit_sim.covariance

with open(f"./data/sol.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(grss_sol, indent=4))

np.savetxt(f"./data/cov.txt", grss_cov, fmt="%30.20e")

# propagation setup

t_end = Time("2041-04-25", scale="tdb", format="iso").tdb.mjd

ki_arriv = 63546.001   

NUM_SAMPLES = int(1e3)
NUM_THREADS = 8

with open(f"./data/sol.json", "r", encoding="utf-8") as f:
    init_sol = json.load(f)

init_cov = np.loadtxt(f"./data/cov.txt")
ng_params = make_nongrav_params(init_sol, nongrav_info)


def generate_samples_dict(sol, cov, num_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    elem_keys = [key for key in sol.keys() if key != "t"]
    sol_arr = np.array([sol[key] for key in elem_keys], dtype=float)

    samples = rng.multivariate_normal(sol_arr, cov, num_samples)

    samples_dict = []
    for i in range(num_samples):
        sample_dict = {"t": float(sol["t"])}
        for j, key in enumerate(elem_keys):
            sample_dict[key] = float(samples[i, j])
        samples_dict.append(sample_dict)

    return samples_dict


RNG = np.random.default_rng(42)

prop_samples_dict = generate_samples_dict(
    init_sol,
    init_cov,
    NUM_SAMPLES,
    rng=RNG,
)

sample_fname = f"./data/samples.json.gz"

with gzip.open(sample_fname, "wt", encoding="utf-8") as file:
    file.write(json.dumps(prop_samples_dict, ensure_ascii=True, indent=4))

print(f"Wrote {NUM_SAMPLES} samples to {sample_fname}")

with gzip.open(sample_fname, "rt", encoding="utf-8") as file:
    prop_samples_dict = json.load(file)

print(type(prop_samples_dict[0]))
print(prop_samples_dict[0])

# GRSS propagation helpers - set up like grss parallel clone propagation but returns state vector 

def make_prop_sim(name, t0, t_final, t_eval=None):
    sim = prop.PropSimulation(
        name=name,
        t0=t0,
        defaultSpiceBodies=440,
        DEkernelPath=utils.default_kernel_path,
    )

    if t_eval is None:
        sim.set_integration_parameters(t_final)
    else:
        sim.set_integration_parameters(t_final, tEval=t_eval)

    return sim


def _is_cometary_solution(sol):
    return all(k in sol for k in ["e", "q", "tp", "om", "w", "i"])


def nongrav_to_dict(ng):
    return {
        "a1": float(ng.a1),
        "a1Est": bool(ng.a1Est),
        "a2": float(ng.a2),
        "a2Est": bool(ng.a2Est),
        "a3": float(ng.a3),
        "a3Est": bool(ng.a3Est),
        "alpha": float(ng.alpha),
        "k": float(ng.k),
        "m": float(ng.m),
        "n": float(ng.n),
        "r0_au": float(ng.r0_au),
    }


def _build_ng_from_clone(clone, ref_ng_dict):
    ng = prop.NongravParameters()

    ng.a1 = float(clone.get("a1", ref_ng_dict["a1"]))
    ng.a1Est = ("a1" in clone) or ref_ng_dict["a1Est"]

    ng.a2 = float(clone.get("a2", ref_ng_dict["a2"]))
    ng.a2Est = ("a2" in clone) or ref_ng_dict["a2Est"]

    ng.a3 = float(clone.get("a3", ref_ng_dict["a3"]))
    ng.a3Est = ("a3" in clone) or ref_ng_dict["a3Est"]

    ng.alpha = ref_ng_dict["alpha"]
    ng.k = ref_ng_dict["k"]
    ng.m = ref_ng_dict["m"]
    ng.n = ref_ng_dict["n"]
    ng.r0_au = ref_ng_dict["r0_au"]

    return ng


def _propagate_one_clone_state(args):
    (
        clone,
        ref_nongrav,
        sim_name,
        de_kernel_path,
        t_final,
        t_eval,
        mass_val,
        radius_val,
    ) = args

    is_cometary = _is_cometary_solution(clone)

    sim = prop.PropSimulation(
        name=sim_name,
        t0=float(clone["t"]),
        defaultSpiceBodies=440,
        DEkernelPath=de_kernel_path,
    )

    if t_eval is None:
        sim.set_integration_parameters(float(t_final))
    else:
        sim.set_integration_parameters(float(t_final), tEval=t_eval)

    ng = _build_ng_from_clone(clone, ref_nongrav)

    if is_cometary:
        body = libgrss.IntegBody(
            "clone",
            float(clone["t"]),
            float(mass_val),
            float(radius_val),
            [
                float(clone["e"]),
                float(clone["q"]),
                float(clone["tp"]),
                float(clone["om"]),
                float(clone["w"]),
                float(clone["i"]),
            ],
            ng,
        )
    else:
        body = libgrss.IntegBody(
            "clone",
            float(clone["t"]),
            float(mass_val),
            float(radius_val),
            [
                float(clone["x"]),
                float(clone["y"]),
                float(clone["z"]),
            ],
            [
                float(clone["vx"]),
                float(clone["vy"]),
                float(clone["vz"]),
            ],
            ng,
        )

    sim.add_integ_body(body)
    sim.integrate()

    return {
        "t": float(sim.t),
        "xInteg": [float(x) for x in sim.xInteg],
        "xIntegEval": (
            [[float(v) for v in row] for row in sim.xIntegEval]
            if t_eval is not None else None
        ),
    }


def propagate_clones_return_states(
    ref_sol,
    ref_nongrav,
    ref_sim,
    clones,
    t_final,
    num_threads=1,
    return_history=False,
    n_eval=25,
    mass_val=0.0,
    radius_val=0.0,
):
    if not clones:
        return []

    t_eval = None
    if return_history:
        t_eval = np.linspace(float(ref_sol["t"]), float(t_final), n_eval).tolist()

    ref_ng_dict = nongrav_to_dict(ref_nongrav)

    worker_args = [
        (
            clone,
            ref_ng_dict,
            f"{ref_sim.name}_clone_{i}",
            utils.default_kernel_path,
            t_final,
            t_eval,
            mass_val,
            radius_val,
        )
        for i, clone in enumerate(clones)
    ]

    start = time.time()
    results = [None] * len(clones)

    if num_threads == 1:
        for i, args in enumerate(worker_args):
            results[i] = _propagate_one_clone_state(args)
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            future_map = {
                ex.submit(_propagate_one_clone_state, args): i
                for i, args in enumerate(worker_args)
            }

            for fut in as_completed(future_map):
                idx = future_map[fut]
                results[idx] = fut.result()

    duration = time.time() - start
    mm = int(duration // 60)
    ss = duration % 60

    print(f"Clone propagation took {mm:02d} minute(s) and {ss:.6f} seconds")

    return results


# Propagate nominal clones to April 24, 2041 (prop_sim)
# Propagate nominal clones to kinetic impact arrival epoch *ki_sim)

prop_sim = make_prop_sim(body_id, init_sol["t"], t_end)
ki_sim = make_prop_sim(f"{body_id}_kinetic_impact", init_sol["t"], ki_arriv)

result_end = propagate_clones_return_states(
    ref_sol=init_sol,
    ref_nongrav=ng_params,
    ref_sim=prop_sim,
    clones=prop_samples_dict,
    t_final=t_end,
    num_threads=NUM_THREADS,
    return_history=False,
    mass_val=mass,
    radius_val=0.0,
)

result_fast = propagate_clones_return_states(
    ref_sol=init_sol,
    ref_nongrav=ng_params,
    ref_sim=ki_sim,
    clones=prop_samples_dict,
    t_final=ki_arriv,
    num_threads=NUM_THREADS,
    return_history=True,
    mass_val=mass,
    radius_val=0.0,
)

print("Finished pre-impact clone propagation.")
print("Example no-intervention end state:", result_end[0])
print("Example kinetic-impact arrival state:", result_fast[0]) #for debug

with open(f"./data/propagated_states_end.json", "w", encoding="utf-8") as f:
    json.dump(result_end, f, indent=2)

with open(f"./data/propagated_states_fast_arrival.json", "w", encoding="utf-8") as f:
    json.dump(result_fast, f, indent=2)

print("Saved pre-impact propagated clone states.")
# ==========================================================
# Kinetic impact uncertainty model
# Separate runs:
#   1. mass-only uncertainty
#   2. beta-only uncertainty
#   3. surface-normal-only uncertainty
# ==========================================================

AU_KM = 149597870.7
SEC_PER_DAY = 86400.0
AU_PER_DAY_TO_KM_PER_S = AU_KM / SEC_PER_DAY
AU_PER_DAY_TO_MM_PER_S = AU_PER_DAY_TO_KM_PER_S * 1.0e6

# Incoming relative velocity vector [au/day]
dv_intercept = np.array([0.0, -0.003848, 0.003007], dtype=float)  # au/day (8.47 km/s)
dv_mag = np.linalg.norm(dv_intercept)
norm_dv = dv_intercept / dv_mag

nominal_v_inf = dv_intercept 

v_inf_hat = nominal_v_inf / np.linalg.norm(nominal_v_inf)
v_inf_mag = np.linalg.norm(nominal_v_inf)
m_sc = 800.0 # spacecraft mass [kg]
# Asteroid mass uncertainty [kg]
M_AST_MIN = 2.82e9
M_AST_MAX = 4.67e9
M_AST_NOM = 3.74e9
# Momentum enhancement uncertainty
BETA_MIN = 1.0
BETA_MAX = 4.5
BETA_NOM = 2.8 
# Surface-normal uncertainty
ALPHA_MAX_DEG = 20.0
if v_inf_hat[1] < 0:
    NORMAL_SIGN = 1.0
else:
    NORMAL_SIGN = -1.0
# Nominal normal direction
N_HAT_NOM = NORMAL_SIGN * v_inf_hat


def random_unit_vector_in_cone(axis_hat, max_angle_deg, rng):
    axis_hat = np.asarray(axis_hat, dtype=float)
    axis_hat /= np.linalg.norm(axis_hat)

    max_angle = np.deg2rad(max_angle_deg)

    cos_alpha = rng.uniform(np.cos(max_angle), 1.0)
    sin_alpha = np.sqrt(1.0 - cos_alpha**2)
    phi = rng.uniform(0.0, 2.0 * np.pi)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis_hat)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = tmp - np.dot(tmp, axis_hat) * axis_hat
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(axis_hat, e1)
    e2 /= np.linalg.norm(e2)

    n_hat = (
        cos_alpha * axis_hat
        + sin_alpha * np.cos(phi) * e1
        + sin_alpha * np.sin(phi) * e2
    )

    return n_hat / np.linalg.norm(n_hat)


def compute_dv_ast(M_ast, beta, n_hat):
    """
    Asteroid velocity change from kinetic impact.
    Units au/day 
    """
    return beta * (m_sc / M_ast) * v_inf_mag * n_hat


def make_dv_metadata(sample_index, case_name, M_ast, beta, n_hat, dv_ast):
    return {
        "sample_index": int(sample_index),
        "case_name": case_name,
        "M_ast_kg": float(M_ast),
        "beta": float(beta),
        "alpha_max_deg": float(ALPHA_MAX_DEG),
        "normal_sign": float(NORMAL_SIGN),
        "n_hat": np.asarray(n_hat, dtype=float).tolist(),
        "dv_au_per_day": np.asarray(dv_ast, dtype=float).tolist(),
        "dv_mag_au_per_day": float(np.linalg.norm(dv_ast)),
        "dv_mag_mm_per_s": float(np.linalg.norm(dv_ast) * AU_PER_DAY_TO_MM_PER_S),
    }


def sample_mass_only_dv(num_samples, rng):
    """
    Only asteroid mass varies.
    beta and normal direction are held fixed to nominal values.
    """
    dv_samples = []
    meta = []

    for i in range(num_samples):
        M_ast = rng.uniform(M_AST_MIN, M_AST_MAX)
        beta = BETA_NOM
        n_hat = N_HAT_NOM.copy()

        dv_ast = compute_dv_ast(M_ast, beta, n_hat)

        dv_samples.append(dv_ast)
        meta.append(make_dv_metadata(i, "mass_only", M_ast, beta, n_hat, dv_ast))

    return np.asarray(dv_samples), meta


def sample_beta_only_dv(num_samples, rng):
    """
    Only beta varies.
    asteroid mass and normal direction are held fixed to nominal values.
    """
    dv_samples = []
    meta = []

    for i in range(num_samples):
        M_ast = M_AST_NOM
        beta = rng.uniform(BETA_MIN, BETA_MAX)
        n_hat = N_HAT_NOM.copy()

        dv_ast = compute_dv_ast(M_ast, beta, n_hat)

        dv_samples.append(dv_ast)
        meta.append(make_dv_metadata(i, "beta_only", M_ast, beta, n_hat, dv_ast))

    return np.asarray(dv_samples), meta


def sample_normal_only_dv(num_samples, rng):
    """
    Only the normal vector direction varies.
    asteroid mass and beta are held fixed to nominal values.
    """
    dv_samples = []
    meta = []

    for i in range(num_samples):
        M_ast = M_AST_NOM
        beta = BETA_NOM

        n_hat = random_unit_vector_in_cone(
            axis_hat=v_inf_hat,
            max_angle_deg=ALPHA_MAX_DEG,
            rng=rng,
        )

        n_hat = NORMAL_SIGN * n_hat

        dv_ast = compute_dv_ast(M_ast, beta, n_hat)

        dv_samples.append(dv_ast)
        meta.append(make_dv_metadata(i, "normal_only", M_ast, beta, n_hat, dv_ast))

    return np.asarray(dv_samples), meta


# Post-impact propagation 

def build_cartesian_integ_body_from_state(
    state,
    t0,
    ng,
    name="body",
    mass_val=0.0,
    radius_val=0.0,
):
    state = np.asarray(state, dtype=float)

    pos = state[:3].tolist()
    vel = state[3:].tolist()

    return libgrss.IntegBody(
        name,
        float(t0),
        float(mass_val),
        float(radius_val),
        pos,
        vel,
        ng,
    )


def propagate_one_cartesian_state(
    state,
    t0,
    t_final,
    ng,
    sim_name,
    t_eval=None,
    mass_val=0.0,
    radius_val=0.0,
):
    sim = prop.PropSimulation(
        name=sim_name,
        t0=float(t0),
        defaultSpiceBodies=440,
        DEkernelPath=utils.default_kernel_path,
    )

    if t_eval is None:
        sim.set_integration_parameters(float(t_final))
    else:
        sim.set_integration_parameters(float(t_final), tEval=t_eval)

    body = build_cartesian_integ_body_from_state(
        state=state,
        t0=t0,
        ng=ng,
        name=sim_name,
        mass_val=mass_val,
        radius_val=radius_val,
    )

    sim.add_integ_body(body)
    sim.integrate()

    return {
        "t": float(sim.t),
        "xInteg": [float(x) for x in sim.xInteg],
        "xIntegEval": (
            [[float(v) for v in row] for row in sim.xIntegEval]
            if t_eval is not None else None
        ),
    }


def _propagate_one_post_impact_state(args):
    (
        item,
        dv_vec,
        t0,
        t_final,
        ref_ng_dict,
        sim_name,
        mass_val,
        radius_val,
    ) = args

    state0 = np.asarray(item["xInteg"], dtype=float).copy()
    state0[3:] += np.asarray(dv_vec, dtype=float)

    ng = prop.NongravParameters()
    ng.a1 = ref_ng_dict["a1"]
    ng.a1Est = ref_ng_dict["a1Est"]
    ng.a2 = ref_ng_dict["a2"]
    ng.a2Est = ref_ng_dict["a2Est"]
    ng.a3 = ref_ng_dict["a3"]
    ng.a3Est = ref_ng_dict["a3Est"]
    ng.alpha = ref_ng_dict["alpha"]
    ng.k = ref_ng_dict["k"]
    ng.m = ref_ng_dict["m"]
    ng.n = ref_ng_dict["n"]
    ng.r0_au = ref_ng_dict["r0_au"]

    return propagate_one_cartesian_state(
        state=state0,
        t0=t0,
        t_final=t_final,
        ng=ng,
        sim_name=sim_name,
        t_eval=None,
        mass_val=mass_val,
        radius_val=radius_val,
    )


def propagate_post_impact_states_parallel(
    saved_results,
    dv_samples,
    t0,
    t_final,
    ref_nongrav,
    num_threads=1,
    mass_val=0.0,
    radius_val=0.0,
    label="post_kinetic_impact",
):
    if len(saved_results) != len(dv_samples):
        raise ValueError(
            f"saved_results has {len(saved_results)} entries but "
            f"dv_samples has {len(dv_samples)} entries."
        )

    ref_ng_dict = nongrav_to_dict(ref_nongrav)

    worker_args = [
        (
            item,
            dv_samples[i],
            t0,
            t_final,
            ref_ng_dict,
            f"{label}_{i}",
            mass_val,
            radius_val,
        )
        for i, item in enumerate(saved_results)
    ]

    start = time.time()
    results = [None] * len(saved_results)

    if num_threads == 1:
        for i, args in enumerate(worker_args):
            results[i] = _propagate_one_post_impact_state(args)
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            future_map = {
                ex.submit(_propagate_one_post_impact_state, args): i
                for i, args in enumerate(worker_args)
            }

            for fut in as_completed(future_map):
                idx = future_map[fut]
                results[idx] = fut.result()

    duration = time.time() - start
    mm = int(duration // 60)
    ss = duration % 60

    print(f"{label} propagation took {mm:02d} minute(s) and {ss:.6f} seconds")

    return results



ng_params_ki = prop.NongravParameters()
ng_params_ki.a1 = ng_params.a1
ng_params_ki.a1Est = ng_params.a1Est
ng_params_ki.a2 = ng_params.a2
ng_params_ki.a2Est = ng_params.a2Est
ng_params_ki.a3 = ng_params.a3
ng_params_ki.a3Est = ng_params.a3Est
ng_params_ki.alpha = ng_params.alpha
ng_params_ki.k = ng_params.k
ng_params_ki.m = ng_params.m
ng_params_ki.n = ng_params.n
ng_params_ki.r0_au = ng_params.r0_au


# Run uncertainty cases 

def run_uncertainty_case(case_name, sampler_func):
    print("\n" + "=" * 70)
    print(f"Running kinetic-impact uncertainty case: {case_name}")
    print("=" * 70)

    dv_samples, deflection_meta = sampler_func(
        num_samples=len(result_fast),
        rng=RNG,
    )

    meta_file = f"./data/kinetic_impact_{case_name}_samples.json"
    state_file = f"./data/post_fast_impact_{case_name}_states.json"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(deflection_meta, f, indent=2)

    print(f"Saved sampled Δv metadata to: {meta_file}")
    print("Example sampled uncertainty:")
    print(deflection_meta[0])

    post_results = propagate_post_impact_states_parallel(
        saved_results=result_fast,
        dv_samples=dv_samples,
        t0=ki_arriv,
        t_final=t_end,
        ref_nongrav=ng_params_ki,
        num_threads=NUM_THREADS,
        mass_val=mass,
        radius_val=0.0,
        label=f"post_kinetic_impact_{case_name}",
    )

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(post_results, f, indent=2)

    print(f"Saved post-impact states to: {state_file}")
    print("Example post-impact state:")
    print(post_results[0])

    return post_results, deflection_meta


post_mass_only_results, mass_only_meta = run_uncertainty_case(
    case_name="mass_only",
    sampler_func=sample_mass_only_dv,
)

post_beta_only_results, beta_only_meta = run_uncertainty_case(
    case_name="beta_only",
    sampler_func=sample_beta_only_dv,
)

post_normal_only_results, normal_only_meta = run_uncertainty_case(
    case_name="normal_only",
    sampler_func=sample_normal_only_dv,
)

print("\nFinished all separate kinetic-impact uncertainty propagations.")