import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
from utils.encoder.audio_encoder import load_audio_model, get_audio_embeddings


def generate_audio_embeddings(
    model_name="facebook/wav2vec2-base-960h",
    stimuli_dir="data/audio_data/ds003020-download/stimuli",
    save_root="filterData/audio/design_matrix",
    tr=2.0,
    device="mps",
    sr_target=16000,
):

    # 初始化保存目录
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    print(f"Model Name: {model_name}")
    processor, model = load_audio_model(model_name, device)
    model.eval()

    
    # 遍历所有 .wav 文件
    wav_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith(".wav")])

    for fname in wav_files:
        audio_path = os.path.join(stimuli_dir, fname)

        # 提取 embedding
        X_layers = get_audio_embeddings(
            audio_path, processor, model,
            device=device, tr=tr, sr_target=sr_target
        )

        X_layers = X_layers.astype(np.float16)

        # 保存路径
        save_name = fname.replace(".wav", ".npy")
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, X_layers)

        print(f"Saved (FP16): {save_path}")

    print("\nFinished all! Saved in:", save_dir)


# 调用
if __name__ == "__main__":
    audio_models = [
        "facebook/data2vec-audio-base",
        "facebook/data2vec-audio-large",

        "facebook/hubert-base-ls960",
        "facebook/hubert-large-ls960-ft",

        "facebook/wav2vec2-base-960h",
        "superb/wav2vec2-base-superb-ks",
        "facebook/wav2vec2-large-xlsr-53",
        "facebook/wav2vec2-xls-r-1b",
        "facebook/wav2vec2-xls-r-300m",
        
        "microsoft/wavlm-base",
        "microsoft/wavlm-base-plus",
        "microsoft/wavlm-large",
    ]

    for model_name in audio_models:
        print("\n====================================")
        print(f"Running model: {model_name}")

        generate_audio_embeddings(
            model_name=model_name,
            device="mps",
            tr=2.0
        )


