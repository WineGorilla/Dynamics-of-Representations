import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# -------------------------
# load eigenvalues
# -------------------------
eigvals = np.load("processed/eigvals/eigvals_vision.npy")


# =========================
# 1 spectral radius
# =========================

rho = np.abs(eigvals)

plt.figure()

plt.hist(
    rho,
    bins=100,
    density=True
)

plt.xlabel("|λ| (spectral radius)")
plt.ylabel("Density")

plt.title("Vision Spectral Radius Distribution")

plt.xlim(0,3)
plt.show()



# =========================
# 2 frequency distribution
# =========================

theta = np.angle(eigvals)

plt.figure()

plt.hist(
    theta,
    bins=100,
    density=True
)

plt.xlabel("angle(λ)")
plt.ylabel("Density")

plt.title("Language Spectral Frequency Distribution")

plt.xlim(-np.pi, np.pi)

plt.show()



# =========================
# 3 eigenvalue density
# =========================

plt.figure(figsize=(6,6))

plt.hist2d(
    eigvals.real,
    eigvals.imag,
    bins=100,
    range=[[-1.5,1.5],[-1.5,1.5]],
    cmap="magma",
    norm=LogNorm()
)

plt.colorbar(label="log density")

# axis
plt.axhline(0,color="white",linewidth=1)
plt.axvline(0,color="white",linewidth=1)

# unit circle
theta_circle = np.linspace(0,2*np.pi,500)
plt.plot(
    np.cos(theta_circle),
    np.sin(theta_circle),
    "--",
    color="white",
    linewidth=2
)

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

plt.xlabel("Re(λ)")
plt.ylabel("Im(λ)")

plt.title("Language DMD Eigenvalue Density")

plt.show()