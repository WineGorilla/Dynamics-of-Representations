import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from core.eigenvalues import collect_one_language_model


language_models = [
    "albert-base-v2",
    "albert-large-v2",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "deberta-base",
    "deberta-large",
    "distilbert-base-uncased",
    "electra-base-discriminator",
    "electra-large-discriminator",
    "roberta-base",
    "roberta-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

os.makedirs("processed/eigvals/language", exist_ok=True)

for model in language_models:
    print(f"Collecting: {model}")
    eigvals = collect_one_language_model(model=model)
    save_path = f"processed/eigvals/language/{model.replace('/', '_')}.npy"
    np.save(save_path, eigvals)
    print(f"  saved → {save_path}  shape={eigvals.shape}")

print("\nDone.")

# all_eigs_language = []

# for model in language_models:
#     eigvals = collect_one_language_model(model=model)
#     all_eigs_language.append(eigvals)

# eigvals_language = np.concatenate(all_eigs_language)

# rho = np.abs(eigvals_language)

# np.save("processed/eigvals/eigvals_language.npy", eigvals_language)






# plt.hist(
#     rho,
#     bins=80,
#     density=True
# )

# plt.xlabel("|λ|")
# plt.ylabel("Density")

# plt.title("Language Spectral Radius Distribution")
# plt.xlim(0, 3)
# plt.show()




# from matplotlib.colors import LogNorm

# plt.figure(figsize=(6,6))

# plt.hist2d(
#     eigvals_language.real,
#     eigvals_language.imag,
#     bins=200,
#     range=[[-1.5,1.5],[-1.5,1.5]],
#     cmap="magma",
#     norm=LogNorm()
# )

# plt.colorbar(label="log density")

# theta = np.linspace(0,2*np.pi,500)

# plt.plot(np.cos(theta), np.sin(theta),"--",color="white",linewidth=2)

# plt.axhline(0,color="white",linewidth=1)
# plt.axvline(0,color="white",linewidth=1)

# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)

# plt.xlabel("Re(λ)")
# plt.ylabel("Im(λ)")

# plt.title("Language DMD Eigenvalue Density")

# plt.show()