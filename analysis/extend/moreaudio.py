import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import gc
import numpy as np
import torch
from tqdm import tqdm
from models.audio_encoder import load_audio, load_audio_model


def generate_audio_embeddings(
    model_name,
    stimuli_dir="data/audio_data/ds003020-download/stimuli",
    save_root="filterData/audio/design_matrix_extra",
    tr=2.0,
    device="mps",
    sr_target=16000,
):
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Model Name: {model_name}")
    processor, model = load_audio_model(model_name, device)
    model.eval()

    def get_audio_embeddings(audio_path, processor, model, device, tr=2.0, sr_target=16000):
        y, sr = load_audio(audio_path, target_sr=sr_target)
        chunk_size = int(sr_target * tr)
        chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

        n_layers = model.config.num_hidden_layers + 1
        layer_accum = [[] for _ in range(n_layers)]

        for chunk in tqdm(chunks, desc=f"{os.path.basename(audio_path)}"):
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            inputs = processor(chunk, sampling_rate=sr_target, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            for l, h in enumerate(hidden_states):
                emb = h.mean(dim=1).squeeze(0).cpu().numpy()
                layer_accum[l].append(emb)

        X = np.stack([np.stack(layer_accum[l], axis=0) for l in range(n_layers)], axis=0)
        return X

    wav_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith(".wav")])

    for fname in wav_files:
        audio_path = os.path.join(stimuli_dir, fname)

        X_layers = get_audio_embeddings(
            audio_path, processor, model,
            device=device, tr=tr, sr_target=sr_target
        )

        X_layers = X_layers.astype(np.float16)

        save_name = fname.replace(".wav", ".npy")
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, X_layers)

        print(f"Saved (FP16): {save_path}")

    print(f"\nDone: {save_dir}")

    # 释放显存
    del model
    del processor
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    extra_audio_models = [
        # ── Wav2Vec2 补充 ──────────────────────────────
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60-self",

        # ── Whisper 系列 ───────────────────────────────
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v2",

        # ── CLAP 系列 ──────────────────────────────────
        "laion/larger_clap_general",
        "laion/larger_clap_music_and_speech",

        # ── UniSpeech-SAT 系列 ─────────────────────────
        "microsoft/unispeech-sat-base",
        "microsoft/unispeech-sat-large",
        "microsoft/unispeech-large-1500h-cv",

        # ── AudioMAE ───────────────────────────────────
        "facebook/audiomae-base",
    ]

    for model_name in extra_audio_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_audio_embeddings(
                model_name=model_name,
                device="mps",
                tr=2.0
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            torch.mps.empty_cache()