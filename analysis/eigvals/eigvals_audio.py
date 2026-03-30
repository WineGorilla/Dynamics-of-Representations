import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import matplotlib.pyplot as plt
from core.eigenvalues import collect_one_audio_model


audio_models = [
    'data2vec-audio-base',
    'data2vec-audio-large',
    'hubert-base-ls960',
    'hubert-large-ls960-ft',
    'wav2vec2-base-960h',
    'wav2vec2-base-superb-ks',
    'wav2vec2-large-xlsr-53',
    'wav2vec2-xls-r-1b',
    'wav2vec2-xls-r-300m',
    'wavlm-base',
    'wavlm-base-plus',
    'wavlm-large',
]


os.makedirs("processed/eigvals/audio", exist_ok=True)

for model in audio_models:
    print(f"Collecting: {model}")
    eigvals = collect_one_audio_model(model=model)
    save_path = f"processed/eigvals/audio/{model.replace('/', '_')}.npy"
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

