"""
统一音频 Embedding 提取脚本（纯音频数据集版）
==============================================
支持模型 (36个):
  Wav2Vec2, HuBERT, WavLM, Data2Vec-Audio,
  UniSpeech, UniSpeech-SAT, SEW, SEW-D,
  Whisper encoder, Wav2Vec2-Conformer, Wav2Vec2-BERT, SUPERB

处理方式:
  每条音频按固定时长 (默认2s) 切chunk，每个chunk取一个embedding
  输出 shape = (n_layers, n_chunks, feat_dim)

输出结构:
  embeddings/audio/{model_tag}/{filename}.npy

用法:
CUDA_VISIBLE_DEVICES=1 python extractnew/extract_audio_embeddings.py \
    --data_root /data/mi2-interns/ruiyu/Dynamics-of-Representations/ESC-50/audio \
    --save_root embeddings \
    --device cuda \
    --chunk_duration 2.0
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
import gc
import argparse
import numpy as np
from tqdm import tqdm

import torch
import librosa
from transformers import (
    AutoProcessor,
    AutoFeatureExtractor,
    AutoModel,
    WhisperModel,
    WhisperFeatureExtractor,
)


# ====================================================================
#  音频文件收集
# ====================================================================
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def collect_audio_paths(data_root):
    """递归收集所有音频文件路径"""
    paths = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    print(f"共找到 {len(paths)} 条音频 (根目录: {data_root})")
    return paths


# ====================================================================
#  音频加载
# ====================================================================
def load_audio(file_path, target_sr=16000):
    waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if waveform_np is None or len(waveform_np) == 0:
        raise ValueError(f"Empty waveform at {file_path}")
    return waveform_np, sr


# ====================================================================
#  模型加载
# ====================================================================
def load_audio_model(model_name, device="cuda"):
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

    model = model.to(device).eval()
    return processor, model


# ====================================================================
#  单条音频提取 (按 chunk 切分)
# ====================================================================
def get_audio_embeddings(audio_path, processor, model, device,
                         chunk_duration=2.0, sr_target=16000):
    """
    对单条音频按 chunk_duration 切分，每个 chunk 提取各层 embedding。
    返回 shape = (n_layers, n_chunks, feat_dim)
    """
    y, sr = load_audio(audio_path, target_sr=sr_target)
    model_type = getattr(model.config, "model_type", "")

    chunk_size = int(sr_target * chunk_duration)
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

    # 确定层数
    if model_type == "whisper":
        n_layers = model.config.encoder_layers + 1
    else:
        n_layers = model.config.num_hidden_layers + 1

    layer_accum = [[] for _ in range(n_layers)]

    for chunk in chunks:
        # pad 不足 chunk_size 的部分
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        if model_type == "whisper":
            inputs = processor(
                chunk, sampling_rate=sr_target, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(inputs["input_features"], output_hidden_states=True)
                hidden_states = outputs.hidden_states
        else:
            inputs = processor(
                chunk, sampling_rate=sr_target, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

        for l, h in enumerate(hidden_states):
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()
            layer_accum[l].append(emb)

    # (n_layers, n_chunks, feat_dim)
    X = np.stack([np.stack(layer_accum[l], axis=0) for l in range(n_layers)], axis=0)
    return X


# ====================================================================
#  模型名称映射 (短名称 → HuggingFace ID)  共36个
# ====================================================================
AUDIO_MODELS = {
    # ── Data2Vec-Audio (4) ──
    "data2vec-audio-base":          "facebook/data2vec-audio-base",
    "data2vec-audio-base-960h":     "facebook/data2vec-audio-base-960h",
    "data2vec-audio-large":         "facebook/data2vec-audio-large",
    "data2vec-audio-large-960h":    "facebook/data2vec-audio-large-960h",

    # ── HuBERT (4) ──
    "hubert-base-ls960":            "facebook/hubert-base-ls960",
    "hubert-base-superb-ks":        "superb/hubert-base-superb-ks",
    "hubert-large-ls960-ft":        "facebook/hubert-large-ls960-ft",
    "hubert-xlarge-ls960-ft":       "facebook/hubert-xlarge-ls960-ft",

    # ── SEW / SEW-D (6) ──
    "sew-d-mid-100k":               "asapp/sew-d-mid-100k",
    "sew-d-small-100k":             "asapp/sew-d-small-100k",
    "sew-d-tiny-100k":              "asapp/sew-d-tiny-100k",
    "sew-mid-100k":                 "asapp/sew-mid-100k",
    "sew-small-100k":               "asapp/sew-small-100k",
    "sew-tiny-100k":                "asapp/sew-tiny-100k",

    # ── UniSpeech / UniSpeech-SAT (4) ──
    "unispeech-large-1500h-cv":     "microsoft/unispeech-large-1500h-cv",
    "unispeech-sat-base":           "microsoft/unispeech-sat-base",
    "unispeech-sat-base-plus":      "microsoft/unispeech-sat-base-plus",
    "unispeech-sat-large":          "microsoft/unispeech-sat-large",

    # ── Wav2Vec2-BERT (1) ──
    "w2v-bert-2.0":                 "facebook/w2v-bert-2.0",

    # ── Wav2Vec2 (10) ──
    "wav2vec2-base":                "facebook/wav2vec2-base",
    "wav2vec2-base-960h":           "facebook/wav2vec2-base-960h",
    "wav2vec2-base-superb-ks":      "superb/wav2vec2-base-superb-ks",
    "wav2vec2-conformer-rel-pos-large":       "facebook/wav2vec2-conformer-rel-pos-large",
    "wav2vec2-conformer-rope-large-960h-ft":  "facebook/wav2vec2-conformer-rope-large-960h-ft",
    "wav2vec2-large":               "facebook/wav2vec2-large",
    "wav2vec2-large-960h":          "facebook/wav2vec2-large-960h",
    "wav2vec2-large-xlsr-53":       "facebook/wav2vec2-large-xlsr-53",
    "wav2vec2-xls-r-1b":            "facebook/wav2vec2-xls-r-1b",
    "wav2vec2-xls-r-300m":          "facebook/wav2vec2-xls-r-300m",

    # ── WavLM (3) ──
    "wavlm-base":                   "microsoft/wavlm-base",
    "wavlm-base-plus":              "microsoft/wavlm-base-plus",
    "wavlm-large":                  "microsoft/wavlm-large",

    # ── Whisper (4) ──
    "whisper-base":                 "openai/whisper-base",
    "whisper-medium":               "openai/whisper-medium",
    "whisper-small":                "openai/whisper-small",
    "whisper-tiny":                 "openai/whisper-tiny",
}


def resolve_model_name(name):
    """短名称 → 完整 HuggingFace ID"""
    return AUDIO_MODELS.get(name, name)


# ====================================================================
#  主流程
# ====================================================================
def run_extraction(
    model_name,
    audio_paths,
    save_root,
    device="cuda",
    chunk_duration=2.0,
    sr_target=16000,
):
    full_name = resolve_model_name(model_name)
    model_tag = full_name.split("/")[-1]

    model_save_dir = os.path.join(save_root, model_tag)
    os.makedirs(model_save_dir, exist_ok=True)

    processor, model = load_audio_model(full_name, device)

    for audio_path in tqdm(audio_paths, desc=f"  {model_tag}", ncols=80, leave=False):
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = os.path.join(model_save_dir, f"{fname}.npy")

        if os.path.exists(save_path):
            continue

        try:
            X = get_audio_embeddings(
                audio_path, processor, model,
                device=device,
                chunk_duration=chunk_duration,
                sr_target=sr_target,
            )
            X = X.astype(np.float16)
            np.save(save_path, X)
        except Exception as e:
            print(f"  ❌ Failed {os.path.basename(audio_path)}: {e}")
            continue

    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ====================================================================
#  CLI
# ====================================================================
ALL_MODEL_NAMES = list(AUDIO_MODELS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一音频 Embedding 提取 (36个模型)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="音频数据集根目录（支持子文件夹或平铺）")
    parser.add_argument("--save_root", type=str, default="embeddings",
                        help="输出根目录，会在下面创建 audio/{model_tag}/ 子目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_duration", type=float, default=2.0,
                        help="每个 chunk 的时长（秒），默认 2.0")
    parser.add_argument("--sr", type=int, default=16000,
                        help="目标采样率，默认 16000")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"要跑的模型名称列表，不指定则跑全部。\n可选: {ALL_MODEL_NAMES}")
    return parser.parse_args()


def main():
    args = parse_args()

    # 收集音频
    audio_paths = collect_audio_paths(args.data_root)
    if len(audio_paths) == 0:
        print("未找到任何音频文件，退出。")
        return

    # 在 save_root 下创建 audio 子目录
    audio_root = os.path.join(args.save_root, "audio")
    os.makedirs(audio_root, exist_ok=True)

    # 确定要跑的模型
    model_list = args.models if args.models else ALL_MODEL_NAMES

    # 保存音频路径索引
    index_path = os.path.join(audio_root, "audio_paths.txt")
    with open(index_path, "w") as f:
        for p in audio_paths:
            f.write(p + "\n")
    print(f"音频索引已保存: {index_path}")

    # 逐模型提取
    for model_name in model_list:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            run_extraction(
                model_name=model_name,
                audio_paths=audio_paths,
                save_root=audio_root,
                device=args.device,
                chunk_duration=args.chunk_duration,
                sr_target=args.sr,
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n全部完成！输出目录: {audio_root}")


if __name__ == "__main__":
    main()