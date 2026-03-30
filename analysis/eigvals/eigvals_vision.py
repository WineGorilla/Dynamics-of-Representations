# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import matplotlib.pyplot as plt
# from core.eigenvalues import collect_one_img_model


# vision_models = [
#     "beit-base-patch16-224-pt22k-ft22k",
#     "beit-large-patch16-224-pt22k-ft22k",
#     "data2vec-vision-base",
#     "data2vec-vision-large",
#     "deit-base-patch16-224",
#     "deit-small-patch16-224",
#     "dino-vitb16",
#     "dino-vits16",
#     "dinov2-base",
#     "dinov2-large",
#     "dinov2-small",
#     "vit-base-patch16-224-in21k",
#     "vit-large-patch16-224-in21k",
#     "vit-mae-base",
#     "vit-mae-large",
#     "vit-msn-base",
#     "vit-msn-large",
# ]

# all_eigs_vision = []

# for model in vision_models:
#     eigvals = collect_one_img_model(model=model)
#     all_eigs_vision.append(eigvals)

# eigvals_vision = np.concatenate(all_eigs_vision)

# rho = np.abs(eigvals_vision)

# np.save("processed/eigvals/eigvals_vision.npy", eigvals_vision)




import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from core.eigenvalues import collect_one_img_model

vision_models = [
    # "beit-base-patch16-224-pt22k-ft22k",
    # "beit-large-patch16-224-pt22k-ft22k",
    # "data2vec-vision-base",
    # "data2vec-vision-large",
    # "deit-base-patch16-224",
    # "deit-small-patch16-224",
    # "dino-vitb16",
    # "dino-vits16",
    # "dinov2-base",
    # "dinov2-large",
    # "dinov2-small",
    # "vit-base-patch16-224-in21k",
    # "vit-large-patch16-224-in21k",
    # "vit-mae-base",
    # "vit-mae-large",
    # "vit-msn-base",
    # "vit-msn-large",
    "resnet50"
]

os.makedirs("processed/eigvals/vision", exist_ok=True)

for model in vision_models:
    print(f"Collecting: {model}")
    eigvals = collect_one_img_model(model=model)
    save_path = f"processed/eigvals/vision/{model.replace('/', '_')}.npy"
    np.save(save_path, eigvals)
    print(f"  saved → {save_path}  shape={eigvals.shape}")

print("\nDone.")



# plt.hist(
#     rho,
#     bins=80,
#     density=True
# )

# plt.xlabel("|λ|")
# plt.ylabel("Density")

# plt.title("Vision Spectral Radius Distribution")
# plt.xlim(0, 3)
# plt.show()





# from matplotlib.colors import LogNorm

# plt.figure(figsize=(6,6))

# plt.hist2d(
#     eigvals_vision.real,
#     eigvals_vision.imag,
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