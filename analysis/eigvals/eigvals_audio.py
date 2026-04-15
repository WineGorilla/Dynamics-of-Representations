import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import matplotlib.pyplot as plt
from core.eigenvalues import collect_one_audio_model

# CUDA_VISIBLE_DEVICES=2 python analysis/eigvals/eigvals_audio.py
audio_models = [
        # ── Wav2Vec2 系列 ─────────────────────────────
        # "wav2vec2-base",
        # "wav2vec2-large",
        # "wav2vec2-base-960h",
        # "wav2vec2-large-960h",
        # "wav2vec2-large-xlsr-53",
        # "wav2vec2-xls-r-300m",
        # "wav2vec2-xls-r-1b",

        # # ── HuBERT 系列 ──────────────────────────────
        # "hubert-base-ls960",
        # "hubert-large-ls960-ft",
        # "hubert-xlarge-ls960-ft",

        # # ── WavLM 系列 ───────────────────────────────
        # "wavlm-base",
        # "wavlm-base-plus",
        # "wavlm-large",

        # # ── Data2Vec-Audio 系列 ───────────────────────
        # "data2vec-audio-base",
        # "data2vec-audio-large",
        # "data2vec-audio-base-960h",
        # "data2vec-audio-large-960h",

        # # ── UniSpeech-SAT 系列 ───────────────────────
        # "unispeech-sat-base",
        # "unispeech-sat-base-plus",
        # "unispeech-sat-large",

        # # ── SEW / SEW-D 系列 ─────────────────────────
        # "sew-tiny-100k",
        # "sew-small-100k",
        # "sew-mid-100k",
        # "sew-d-tiny-100k",
        # "sew-d-small-100k",
        # "sew-d-mid-100k",

        # # ── Whisper 系列（只用 encoder）────────────────
        # "whisper-tiny",
        # "whisper-base",
        # "whisper-small",
        # "whisper-medium",

        # # ── SUPERB 系列 ──────────────────────────────
        # "wav2vec2-base-superb-ks",
        # "hubert-base-superb-ks",
        "wav2vec2-conformer-rel-pos-large",
        "wav2vec2-conformer-rope-large-960h-ft",

        # ── Wav2Vec2-BERT 系列（4.5M 小时预训练）─────────
        "w2v-bert-2.0",

        # ── UniSpeech 系列（区别于 UniSpeech-SAT）────────
        "unispeech-large-1500h-cv",
        
]


os.makedirs("processed_new/eigvals/audio", exist_ok=True)

for model in audio_models:
    print(f"Collecting: {model}")
    eigvals = collect_one_audio_model(model=model)
    save_path = f"processed_new/eigvals/audio/{model.replace('/', '_')}.npy"
    np.save(save_path, eigvals)
    print(f"  saved → {save_path}  shape={eigvals.shape}")

print("\nDone.")



# all_eigs_audio = []

# for model in audio_models:
#     eigvals = collect_one_audio_model(model=model)
#     all_eigs_audio.append(eigvals)

# eigvals_audio = np.concatenate(all_eigs_audio)

# rho = np.abs(eigvals_audio)

# np.save("processed/eigvals/eigvals_audio.npy", eigvals_audio)

