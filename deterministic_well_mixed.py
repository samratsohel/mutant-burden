import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator, NullFormatter

# =====================================================
# FIGURE (SI.2)
# =====================================================
# This script generates a three-panel figure showing:
#
# (a) Qualitative classification of the dependence of
#     mutant burden M_N on the death-to-birth ratio δ
#
# (b) Relative variation of M_N across δ values
#
# (c) Scaling behavior of normalized mutant burden
#     M_N(δ)/M_N(0) as a function of system size N
# =====================================================

# =====================================================
# GRID_SIZE controls the resolution of the phase diagrams.
# =====================================================
GRID_SIZE = 500


# =====================================================
# GLOBAL MATPLOTLIB STYLE
# =====================================================
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 30,
    "axes.titlesize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.linewidth": 1.5,
})


# =====================================================
# DETERMINISTIC MODEL
# =====================================================
# x(t) : wild-type population
# y(t) : mutant population
#
# The model assumes exponential growth with mutation.
# =====================================================

def x(delta, s1, s2, u, t):
    """
    Wild-type population dynamics.
    """
    return np.exp((1 - delta - u) * t)


def y(delta, s1, s2, u, t):
    """
    Mutant population dynamics.

    Includes the special neutral-like case where
    the denominator approaches zero.
    """

    z1 = x(delta, s1, s2, u, t)

    denom = (s1 + s2 * delta - u)

    # Regularized limit when denominator ~ 0
    if np.isclose(denom, 0):
        return z1 * u * t

    z2 = u / denom
    z3 = 1 - np.exp(-denom * t)

    return z1 * z2 * z3


# =====================================================
# SOLVE FOR TERMINATION TIME
# =====================================================
# The stopping time t satisfies:
#
#     x(t) + y(t) = N
#
# i.e., the total population reaches size N.
# =====================================================

def solve_t(delta, s1, s2, u, N):

    def eq(t):
        return (
            x(delta, s1, s2, u, t)
            + y(delta, s1, s2, u, t)
            - N
        )

    # Initial bracket for Brent solver
    t_low = 0.0
    t_high = 1.0

    # Expand upper bracket until root is enclosed
    while eq(t_high) < 0:

        t_high *= 2

        # Safety cutoff
        if t_high > 1e12:
            return None

    # Root finding using Brent's method
    sol = root_scalar(
        eq,
        bracket=[t_low, t_high],
        method='brentq'
    )

    return sol.root if sol.converged else None


# =====================================================
# CURVE CLASSIFICATION
# =====================================================
# Classification of M_N(δ):
#
#   0 : monotonically decreasing
#   1 : non-monotonic
#   2 : monotonically increasing
# =====================================================

def classify_curve(M_vals):

    M = np.array(M_vals)

    # Remove invalid values
    M = M[~np.isnan(M)]

    # Not enough points to classify reliably
    if len(M) < 3:
        return 1

    dM = np.diff(M)

    if np.all(dM > 0):
        return 2

    elif np.all(dM < 0):
        return 0

    else:
        return 1


# =====================================================
# FINITE-N TAKEOVER THEORY
# =====================================================
# These functions describe stochastic takeover
# probabilities in a finite population.
# =====================================================

def qX_finite(delta, N, u):
    """
    Wild-type extinction probability.
    """

    R = delta / (1 - u)

    # Neutral limit
    if np.isclose(R, 1.0):
        return (N - 1) / N

    return (R - R**N) / (1 - R**N)


def rhoY_finite(delta, N, s1, s2):
    """
    Mutant fixation probability.
    """

    R = (delta * (1 + s2)) / (1 - s1)

    # Neutral limit
    if np.isclose(R, 1.0):
        return 1.0 / N

    return (1 - R) / (1 - R**N)


def Lambda_ext(delta, u):
    """
    Expected mutant production before extinction.
    """

    if (1 - u - delta) <= 0:
        return np.inf

    return u / (1 - u - delta)


def P_takeover(delta, N, u, s1, s2):
    """
    Total probability of mutant takeover.
    """

    qx = qX_finite(delta, N, u)

    rho = rhoY_finite(delta, N, s1, s2)

    Lambda = Lambda_ext(delta, u)

    num = qx * (1 - np.exp(-rho * Lambda))

    den = 1 - qx * np.exp(-rho * Lambda)

    return num / den if den > 0 else 0.0


# =====================================================
# EFFECTIVE MUTANT BURDEN
# =====================================================
# Computes the expected mutant burden M_N.
# =====================================================

def compute_MN(delta, s1, s2, u, N):

    # Solve stopping time
    t = solve_t(delta, s1, s2, u, N)

    if t is None:
        return np.nan

    # Deterministic mutant burden
    M_det = y(delta, s1, s2, u, t)

    # Stochastic takeover probability
    P = P_takeover(delta, N, u, s1, s2)

    # Full stochastic expectation
    z_stoch = (1 - P) * M_det + P * N

    # ODE-only contribution
    z_ODE = M_det

    # Current implementation returns deterministic part
    return z_ODE


# =====================================================
# MODEL PARAMETERS
# =====================================================

u = 1e-7          # mutation rate

s1 = 0.0          # birth disadvantage

# δ values scanned in the analysis
delta_vals = np.linspace(0, 1 - u, 40)


# =====================================================
# PARAMETER GRID FOR PHASE DIAGRAMS
# =====================================================

# System size range
N_grid = np.logspace(0, 25, GRID_SIZE)

# Absolute death disadvantage values
s2_abs_vals = np.logspace(-3.0, -0.5, GRID_SIZE)

s2_vals = s2_abs_vals

# Arrays storing results
phase = np.zeros((GRID_SIZE, GRID_SIZE))

rel_change = np.zeros((GRID_SIZE, GRID_SIZE))


# =====================================================
# COMPUTE PHASE DIAGRAMS
# =====================================================
# For every (N, |s2|) pair:
#
#   1. Compute M_N(δ)
#   2. Classify curve shape
#   3. Measure relative variation
# =====================================================

for i, s2 in enumerate(s2_vals):

    print(i, s2)

    for j, N in enumerate(N_grid):

        # Compute full curve M_N(δ)
        M_delta = np.array([
            compute_MN(delta, s1, s2, u, N)
            for delta in delta_vals
        ])

        # Phase classification
        phase[i, j] = classify_curve(M_delta)

        valid = M_delta[~np.isnan(M_delta)]

        # Relative variation across δ
        if len(valid) > 2:

            rel_change[i, j] = (
                np.max(valid)
                - np.min(valid)
            ) / np.max(valid)

        else:

            rel_change[i, j] = 1e-12


# =====================================================
# CLEAN DATA FOR LOGARITHMIC COLOR SCALE
# =====================================================

rel_change = np.nan_to_num(rel_change, nan=1e-12)

rel_change[rel_change <= 0] = 1e-12


# =====================================================
# COLOR SCALE NORMALIZATION
# =====================================================

nonzero = rel_change[rel_change > 0]

vmin = max(np.percentile(nonzero, 5), 1e-8)

vmax = min(np.percentile(nonzero, 95), 1)


# =====================================================
# PANEL (c): SCALING CURVES
# =====================================================

N_plot = np.logspace(0, 25, GRID_SIZE)

deltas_plot = np.arange(0.0, 1.0, 0.2)

s2_fixed = 10**(-1.5) # death disadvantage

# Dictionary storing normalized curves
MN_curves = {delta: [] for delta in deltas_plot}

for N in N_plot:

    # Reference value at δ = 0
    M0 = compute_MN(
        0.0,
        s1,
        s2_fixed,
        u,
        N
    )

    for delta in deltas_plot:

        Mt = compute_MN(
            delta,
            s1,
            s2_fixed,
            u,
            N
        )

        # Normalized mutant burden
        if M0 > 0:
            MN_curves[delta].append(Mt / M0)

        else:
            MN_curves[delta].append(np.nan)


# =====================================================
# CREATE FIGURE
# =====================================================

fig = plt.figure(figsize=(18, 5))


# =====================================================
# AXES LAYOUT
# =====================================================

# Panel (a)
ax1 = fig.add_axes([0.06, 0.15, 0.25, 0.72])

# Panel (b)
ax2 = fig.add_axes([0.38, 0.15, 0.25, 0.72])

# Shared colorbar
cax = fig.add_axes([0.64, 0.15, 0.015, 0.72])

# Panel (c)
ax3 = fig.add_axes([0.81, 0.15, 0.22, 0.72])


# =====================================================
# PANEL (a): PHASE CLASSIFICATION
# =====================================================

im1 = ax1.imshow(
    np.flipud(phase),

    extent=[0, 25, -3, -0.5],

    aspect='auto',

    cmap=mpl.colors.ListedColormap([
        "gray",      # decreasing
        "#66C266",   # non-monotonic
        "#4C72B0"    # increasing
    ])
)

ax1.set_xlabel(r'$N$')

ax1.set_ylabel(r'$|s_2|$')


# -----------------------------------------------------
# X-axis ticks
# -----------------------------------------------------
ax1.set_xticks([0, 5, 10, 15, 20, 25])

ax1.set_xticklabels([
    r'$10^{0}$',
    r'$10^{5}$',
    r'$10^{10}$',
    r'$10^{15}$',
    r'$10^{20}$',
    r'$10^{25}$'
])


# -----------------------------------------------------
# Y-axis ticks
# -----------------------------------------------------
ax1.set_yticks([-3, -2, -1, -0.5])

ax1.set_yticklabels([
    r'$10^{-3}$',
    r'$10^{-2}$',
    r'$10^{-1}$',
    r'$10^{-0.5}$'
])


# Reference horizontal line
ax1.axhline(
    -1.5,
    color='black',
    linestyle='--',
    linewidth=2
)


# =====================================================
# PANEL (b): RELATIVE VARIATION
# =====================================================

im2 = ax2.imshow(
    np.flipud(rel_change),

    extent=[0, 25, -3, -0.5],

    aspect='auto',

    cmap='viridis',

    norm=LogNorm(
        vmin=vmin,
        vmax=vmax
    )
)

ax2.set_xlabel(r'$N$')


# -----------------------------------------------------
# X-axis ticks
# -----------------------------------------------------
ax2.set_xticks([0, 5, 10, 15, 20, 25])

ax2.set_xticklabels([
    r'$10^{0}$',
    r'$10^{5}$',
    r'$10^{10}$',
    r'$10^{15}$',
    r'$10^{20}$',
    r'$10^{25}$'
])


# -----------------------------------------------------
# Y-axis ticks
# -----------------------------------------------------
ax2.set_yticks([-3, -2, -1, -0.5])

ax2.set_yticklabels([
    r'$10^{-3}$',
    r'$10^{-2}$',
    r'$10^{-1}$',
    r'$10^{-0.5}$'
])


# =====================================================
# COLORBAR
# =====================================================

cb = fig.colorbar(im2, cax=cax)

tick_vals = np.logspace(
    np.log10(vmin),
    np.log10(vmax),
    7
)

cb.set_ticks(tick_vals)

tick_labels = []

for val in tick_vals:

    exponent = int(np.floor(np.log10(val)))

    mantissa = val / (10**exponent)

    tick_labels.append(
        rf"${mantissa:.1f}\times10^{{{exponent}}}$"
    )

cb.set_ticklabels(tick_labels)

# Remove minor ticks for cleaner appearance
cb.ax.minorticks_off()

cb.ax.yaxis.set_minor_locator(NullLocator())
cb.ax.yaxis.set_minor_formatter(NullFormatter())

cb.ax.tick_params(labelsize=16)


# =====================================================
# PANEL (c): NORMALIZED MUTANT BURDEN
# =====================================================

colors = plt.cm.viridis(
    np.linspace(0, 1, len(deltas_plot))
)

for delta, c in zip(deltas_plot, colors):

    ax3.plot(
        N_plot,
        MN_curves[delta],
        color=c,
        linewidth=2.5,
        label=rf'$\delta={delta:.1f}$'
    )

# Logarithmic axes
ax3.set_xscale('log')

ax3.set_yscale('log')

ax3.set_xlabel(r'$N$')

ax3.set_ylabel(
    r'$M_N(\delta)/M_N(0)$'
)


# =====================================================
# LEGEND
# =====================================================

ax3.legend(
    frameon=False,
    fontsize=18,
    ncol=1
)


# =====================================================
# PANEL LABELS
# =====================================================

ax1.text(
    0.02,
    1.05,
    r'(a)',
    transform=ax1.transAxes,
    fontsize=28,
    fontweight='bold'
)

ax2.text(
    0.02,
    1.05,
    r'(b)',
    transform=ax2.transAxes,
    fontsize=28,
    fontweight='bold'
)

ax3.text(
    0.02,
    1.05,
    r'(c)',
    transform=ax3.transAxes,
    fontsize=28,
    fontweight='bold'
)


# =====================================================
# SAVE FIGURE
# =====================================================
# Output is saved as a high-resolution PDF 
# =====================================================

plt.savefig(
    "FIGSI2.pdf",
    dpi=300,
    bbox_inches='tight'
)

plt.show()