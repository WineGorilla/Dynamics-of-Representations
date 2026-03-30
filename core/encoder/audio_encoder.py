import torch
import numpy as np
from transformers import AutoProcessor, AutoFeatureExtractor, AutoModel
from tqdm import tqdm
import librosa
import os

def load_audio_model(model_name="facebook/wav2vec2-base-960h", device="mps"):
    print(f"加载音频模型: {model_name}")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    return processor, model

def load_audio(file_path, target_sr=16000):
    """读取音频文件并重采样为目标采样率"""
    waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if waveform_np is None or len(waveform_np) == 0:
        raise ValueError(f"Empty waveform at {file_path}")
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    return waveform, sr


# 计算embedding
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