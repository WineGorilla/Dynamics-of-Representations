import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from core.eigenvalues import collect_one_language_model

# CUDA_VISIBLE_DEVICES=2 python analysis/eigvals/eigvals_language.py
language_models = [
        # ── BERT 系列 ────────────────────────────────
        # "bert-base-uncased",
        # "bert-base-cased",
        # "bert-large-uncased",
        # "bert-large-cased",
        # "bert-base-multilingual-cased",

        # # ── RoBERTa 系列 ─────────────────────────────
        # "roberta-base",
        # "roberta-large",

        # # ── ALBERT 系列 ──────────────────────────────
        # "albert-base-v2",
        # "albert-large-v2",
        # "albert-xlarge-v2",

        # # ── DeBERTa 系列 ─────────────────────────────
        # "deberta-base",
        # "deberta-large",


        # # ── DistilBERT 系列 ──────────────────────────
        # "distilbert-base-uncased",
        # "distilbert-base-multilingual-cased",

        # # ── ELECTRA 系列 ─────────────────────────────
        # "electra-small-discriminator",
        # "electra-base-discriminator",
        # "electra-large-discriminator",

        # # ── XLM-RoBERTa 系列 ─────────────────────────
        # "xlm-roberta-base",
        # "xlm-roberta-large",

        # # ── XLNet 系列 ───────────────────────────────
        # "xlnet-base-cased",
        # "xlnet-large-cased",

        # # ── GPT2 系列 ────────────────────────────────
        # "gpt2",
        # "gpt2-medium",
        # "gpt2-large",

        # # ── T5 encoder 系列 ──────────────────────────
        # "t5-small",

        # ── ERNIE 系列 ───────────────────────────────
        # "ernie-2.0-base-en",
        # "ernie-2.0-large-en",

        # # ── Funnel Transformer 系列 ──────────────────
        # "mpnet-base",
        # "all-MiniLM-L6-v2",
        # "all-mpnet-base-v2",

                # ── SqueezeBERT ──────────────────────────────
        # "squeezebert-uncased",

        # # ── ConvBERT 系列 ────────────────────────────
        # "conv-bert-base",
        # "conv-bert-medium-small",


        # # ── I-BERT（量化友好）────────────────────────
        # "ibert-roberta-base",

        # # ── Data2Vec-Text 系列 ───────────────────────
        # "data2vec-text-base",

        # # ── CamemBERT（法语，但架构通用）─────────────
        # "camembert-base",

        # # ── RemBERT ──────────────────────────────────
        # "rembert",

        # # ── DistilRoBERTa ────────────────────────────
        # "distilroberta-base",
        "MiniLM-L6-H384-uncased"

]

os.makedirs("processed_new/eigvals/language", exist_ok=True)

for model in language_models:
    print(f"Collecting: {model}")
    eigvals = collect_one_language_model(model=model)
    save_path = f"processed_new/eigvals/language/{model.replace('/', '_')}.npy"
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