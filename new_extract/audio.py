import torch
import numpy as np
import librosa
import os
import gc
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoFeatureExtractor, AutoModel,
    WhisperModel, WhisperFeatureExtractor
)
#CUDA_VISIBLE_DEVICES=2 python new_extract/audio.py

def load_audio_model(model_name="facebook/wav2vec2-base-960h", device="mps"):
    print(f"加载音频模型: {model_name}")
    model_lower = model_name.lower()

    if "whisper" in model_lower:
        processor = WhisperFeatureExtractor.from_pretrained(model_name)
        full_model = WhisperModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.encoder
        model.config = full_model.config
        model.config.model_type = "whisper"
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device)
    model.eval()
    return processor, model


def load_audio(file_path, target_sr=16000):
    """读取音频文件并重采样为目标采样率"""
    waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if waveform_np is None or len(waveform_np) == 0:
        raise ValueError(f"Empty waveform at {file_path}")
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    return waveform, sr


def get_audio_embeddings(audio_path, processor, model, device, tr=2.0, sr_target=16000):
    y, sr = load_audio(audio_path, target_sr=sr_target)
    model_type = getattr(model.config, "model_type", "")

    chunk_size = int(sr_target * tr)
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

    if model_type == "whisper":
        n_layers = model.config.encoder_layers + 1
    else:
        n_layers = model.config.num_hidden_layers + 1

    layer_accum = [[] for _ in range(n_layers)]

    for chunk in chunks:
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        if model_type == "whisper":
            inputs = processor(
                chunk, sampling_rate=sr_target,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(
                    inputs["input_features"],
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
        else:
            inputs = processor(chunk, sampling_rate=sr_target, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

        for l, h in enumerate(hidden_states):
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()
            layer_accum[l].append(emb)

    X = np.stack([np.stack(layer_accum[l], axis=0) for l in range(n_layers)], axis=0)
    return X


def generate_audio_embeddings(
    model_name="facebook/wav2vec2-base-960h",
    stimuli_dir="data/audio_data/stimuli",
    save_root="filterData/audio/design_matrix",
    tr=2.0,
    device="cuda",
    sr_target=16000,
):
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    processor, model = load_audio_model(model_name, device)

    wav_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith(".wav")])

    for fname in tqdm(wav_files, desc=f"{model_tag}", ncols=80):
        audio_path = os.path.join(stimuli_dir, fname)
        try:
            X_layers = get_audio_embeddings(
                audio_path, processor, model,
                device=device, tr=tr, sr_target=sr_target
            )
            X_layers = X_layers.astype(np.float16)

            save_name = fname.replace(".wav", ".npy")
            save_path = os.path.join(save_dir, save_name)
            np.save(save_path, X_layers)
        except Exception as e:
            print(f"  ❌ Failed {fname}: {e}")
            continue

    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll Done! Saved in: {save_dir}")


if __name__ == "__main__":
    audio_models = [
        # ── Wav2Vec2 系列 ─────────────────────────────
        # "facebook/wav2vec2-base",
        # "facebook/wav2vec2-large",
        # "facebook/wav2vec2-base-960h",
        # "facebook/wav2vec2-large-960h",
        # "facebook/wav2vec2-large-xlsr-53",
        # "facebook/wav2vec2-xls-r-300m",
        # "facebook/wav2vec2-xls-r-1b",

        # # ── HuBERT 系列 ──────────────────────────────
        # "facebook/hubert-base-ls960",
        # "facebook/hubert-large-ls960-ft",
        # "facebook/hubert-xlarge-ls960-ft",

        # # ── WavLM 系列 ───────────────────────────────
        # "microsoft/wavlm-base",
        # "microsoft/wavlm-base-plus",
        # "microsoft/wavlm-large",

        # # ── Data2Vec-Audio 系列 ───────────────────────
        # "facebook/data2vec-audio-base",
        # "facebook/data2vec-audio-large",
        # "facebook/data2vec-audio-base-960h",
        # "facebook/data2vec-audio-large-960h",

        # # ── UniSpeech-SAT 系列 ───────────────────────
        # "microsoft/unispeech-sat-base",
        # "microsoft/unispeech-sat-base-plus",
        # "microsoft/unispeech-sat-large",

        # # ── SEW / SEW-D 系列 ─────────────────────────
        # "asapp/sew-tiny-100k",
        # "asapp/sew-small-100k",
        # "asapp/sew-mid-100k",
        # "asapp/sew-d-tiny-100k",
        # "asapp/sew-d-small-100k",
        # "asapp/sew-d-mid-100k",

        # # ── Whisper 系列（只用 encoder）────────────────
        # "openai/whisper-tiny",
        # "openai/whisper-base",
        # "openai/whisper-small",
        # "openai/whisper-medium",

        # # ── SUPERB 系列 ──────────────────────────────
        # "superb/wav2vec2-base-superb-ks",
        # "superb/hubert-base-superb-ks",
        # ── Wav2Vec2-Conformer 系列（Conformer 替代 Attention）──
        "facebook/wav2vec2-conformer-rel-pos-large",
        "facebook/wav2vec2-conformer-rope-large-960h-ft",

        # ── Wav2Vec2-BERT 系列（4.5M 小时预训练）─────────
        "facebook/w2v-bert-2.0",

        # ── UniSpeech 系列（区别于 UniSpeech-SAT）────────
        "microsoft/unispeech-large-1500h-cv",
    ]

    for model_name in audio_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_audio_embeddings(
                model_name=model_name,
                device="cuda",
                tr=2.0
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()