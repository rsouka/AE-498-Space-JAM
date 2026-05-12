import numpy as np
import matplotlib.pyplot as plt

# Constants
 
AUDAY_TO_kMPS = (1.496e11 / 86400.0) / 1000.0

dv_intercept = np.array([0.0, -0.003848,  0.003007]) #au/day #8.47 km/s

dv_mag = np.linalg.norm(dv_intercept) 
norm_dv = dv_intercept / dv_mag

V_inf_nom = dv_intercept * AUDAY_TO_kMPS
n_hat_nom = V_inf_nom / np.linalg.norm(V_inf_nom) 
N_CASES = 1000
RNG = np.random.default_rng(42)

m_sc = 800  # impactor mass [kg]

V_inf_nom = dv_intercept * AUDAY_TO_kMPS 
n_hat_nom = V_inf_nom / np.linalg.norm(V_inf_nom)

M_ast_min = 2.82e9
M_ast_max = 4.67e9

beta_min = 1.5
beta_max = 4.5

alpha_max_deg = 20.0
alpha_max = np.radians(alpha_max_deg)

AU_DAY_TO_MM_S = (1.496e11 / 86400.0) * 1000.0


# Helper functions

def perp_basis(n):
    n = n / np.linalg.norm(n)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    t1 = np.cross(ref, n)
    t1 /= np.linalg.norm(t1)

    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)

    return t1, t2


def compute_delta_v(M_ast, beta, V_inf, n_hat, m_sc):
    V_dot_n = np.einsum("j,ij->i", V_inf, n_hat)

    dv = (m_sc / M_ast)[:, None] * (
        V_inf[None, :]
        + ((beta - 1.0) * V_dot_n)[:, None] * n_hat
    )

    return dv


def efficiency_from_angle(alpha_rad, beta):
    return np.sqrt(
        1.0 + (beta**2 - 1.0) * np.cos(alpha_rad)**2
    ) / beta


# Monte Carlo sampling

M_ast = RNG.uniform(M_ast_min, M_ast_max, N_CASES)
beta = RNG.uniform(beta_min, beta_max, N_CASES)

# Angular surface-normal uncertainty
alpha = RNG.uniform(0.0, alpha_max, N_CASES)
phi = RNG.uniform(0.0, 2.0 * np.pi, N_CASES)

t1, t2 = perp_basis(n_hat_nom)

n_hat_mc = np.zeros((N_CASES, 3))

for i in range(N_CASES):
    tangent_dir = np.cos(phi[i]) * t1 + np.sin(phi[i]) * t2

    n_hat_mc[i] = (
        np.cos(alpha[i]) * n_hat_nom
        + np.sin(alpha[i]) * tangent_dir
    )

n_hat_mc /= np.linalg.norm(n_hat_mc, axis=1)[:, None]


# Compute delta V 

dv_ki = compute_delta_v(
    M_ast=M_ast,
    beta=beta,
    V_inf=V_inf_nom,
    n_hat=n_hat_mc,
    m_sc=m_sc,
)

dv_mag_km_s = np.linalg.norm(dv_ki, axis=1)
dv_mag_mm_s = dv_mag_km_s * 1000000.0

eta_angle = efficiency_from_angle(alpha, beta)
loss_angle_pct = 100.0 * (1.0 - eta_angle)

print("ΔV magnitude statistics [mm/s]:")
print(f"  Mean : {dv_mag_mm_s.mean():.3f}")
print(f"  Std  : {dv_mag_mm_s.std():.3f}")
print(f"  Min  : {dv_mag_mm_s.min():.3f}")
print(f"  Max  : {dv_mag_mm_s.max():.3f}")
print(f"  5th percentile  : {np.percentile(dv_mag_mm_s, 5):.3f}")
print(f"  95th percentile : {np.percentile(dv_mag_mm_s, 95):.3f}")


# Plot histogram 

plt.figure(figsize=(7.5, 4.2))

plt.hist(
    dv_mag_mm_s,
    bins=40,
    edgecolor="k",
    alpha=0.75,
)

plt.axvline(
    np.mean(dv_mag_mm_s),
    ls="--",
    lw=2,
    label="mean",
)

plt.axvline(
    np.percentile(dv_mag_mm_s, 5),
    ls=":",
    lw=2,
    label="5–95% range",
)

plt.axvline(
    np.percentile(dv_mag_mm_s, 95),
    ls=":",
    lw=2,
)

plt.xlabel(r"$|\Delta V|$ [mm/s]")
plt.ylabel("Monte Carlo cases")
plt.title(r"Monte Carlo distribution of achieved $\Delta V$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Plot scatter plot 

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
tickfont = 14
axs[0].tick_params(axis="both", which="major", labelsize=tickfont)
axs[1].tick_params(axis="both", which="major", labelsize=tickfont)
axs[2].tick_params(axis="both", which="major", labelsize=tickfont)
axs[0].scatter(M_ast / 1e9, dv_mag_mm_s, s=14, alpha=0.4)
axs[0].set_xlabel(r"Asteroid Mass [$10^9$ kg]", fontsize = 14)
axs[0].set_ylabel(r"$|\Delta V|$ [mm/s]")
axs[0].set_title("Mass Uncertainty", fontsize = 18)

axs[1].scatter(beta, dv_mag_mm_s, s=14, alpha=0.4)
axs[1].set_xlabel(r"$\beta$", fontsize = 14)
axs[1].set_title(r"$\beta$ Uncertainty", fontsize = 18)

axs[2].scatter(np.degrees(alpha), dv_mag_mm_s, s=14, alpha=0.4)
axs[2].set_xlabel(r"Normal Deviation Angle $\alpha$ [deg]", fontsize = 14)
axs[2].set_title("Surface-Normal Uncertainty", fontsize = 18)

for ax in axs:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()