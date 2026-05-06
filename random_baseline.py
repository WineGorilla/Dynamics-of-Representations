"""
Random Baseline: 高斯噪声注入实验
==================================
为三个模态生成随机输入，提取 embedding + DMD 特征值，
验证模态间 |λ| 差异是否来自数据结构而非模型架构。

随机输入:
  - Vision:   1000 张 224x224 高斯随机图片
  - Audio:    1000 条 2s 高斯白噪声
  - Language: 1000 个随机词 (从词表随机抽取)

输出:
  embeddings/random_vision/{model}.npy
  embeddings/random_audio/{model}/*.npy
  embeddings/random_language/{model}/*.npy
  neweigvals/random_vision/{model}.npy
  neweigvals/random_audio/{model}.npy
  neweigvals/random_language/{model}.npy

用法:
  CUDA_VISIBLE_DEVICES=0 python random_baseline.py --step all --device cuda
  CUDA_VISIBLE_DEVICES=0 python random_baseline.py --step extract --modality vision
  python random_baseline.py --step eigvals
  python random_baseline.py --step analysis
"""

import sys
import os
import gc
import argparse
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image

# ─── Vision ───
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoImageProcessor,
    CLIPModel, SamModel,
)
from torchvision import models, transforms

# ─── Audio ───
from transformers import (
    AutoProcessor,
    WhisperModel, WhisperFeatureExtractor,
)

# ─── Language ───
from transformers import AutoTokenizer, T5EncoderModel

# ─── DMD ───
from core.dmd import compute_dmd_eigenvalues


# ####################################################################
#  随机数据生成
# ####################################################################

def generate_random_images(n=1000, size=224, seed=42):
    """生成 n 张高斯随机 PIL 图片"""
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(n):
        # N(0.5, 0.25) clipped to [0, 1], then scale to [0, 255]
        pixels = rng.randn(size, size, 3).astype(np.float32) * 0.25 + 0.5
        pixels = np.clip(pixels, 0, 1)
        pixels = (pixels * 255).astype(np.uint8)
        images.append(Image.fromarray(pixels, 'RGB'))
    return images


def generate_random_audio(n=1000, duration=2.0, sr=16000, seed=42):
    """生成 n 条高斯白噪声音频 (numpy arrays)"""
    rng = np.random.RandomState(seed)
    chunk_size = int(sr * duration)
    audios = []
    for _ in range(n):
        audio = rng.randn(chunk_size).astype(np.float32) * 0.1
        audios.append(audio)
    return audios


def generate_random_words(tokenizer, n=1000, seed=42):
    """从词表随机抽取 n 个词"""
    rng = np.random.RandomState(seed)
    vocab = list(tokenizer.get_vocab().keys())
    # 过滤掉特殊 token 和太短的
    vocab = [w for w in vocab if len(w) > 1 and not w.startswith('[') 
             and not w.startswith('<') and not w.startswith('##')]
    if len(vocab) == 0:
        vocab = list(tokenizer.get_vocab().keys())
    indices = rng.choice(len(vocab), size=n, replace=True)
    return [vocab[i] for i in indices]


# ####################################################################
#  共用工具
# ####################################################################

def random_project(X, target_dim, random_state=42):
    d = X.shape[1]
    if d == target_dim:
        return X.astype(np.float32)
    rng = np.random.RandomState(random_state)
    R = rng.randn(d, target_dim).astype(np.float32) / np.sqrt(target_dim)
    return X.astype(np.float32) @ R


def align_layer_dims(X_layers):
    feat_dims = [X.shape[1] for X in X_layers]
    target_dim = min(feat_dims)
    if len(set(feat_dims)) > 1:
        X_layers = [random_project(X, target_dim) for X in X_layers]
    return X_layers, target_dim


# ####################################################################
#  Vision 提取 (from PIL images directly)
# ####################################################################

NO_CLS_TYPES = {"swin", "sam", "siglip"}

VIT_MODELS = {
    "dinov2-small": "facebook/dinov2-small", "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dino-vitb16": "facebook/dino-vitb16", "dino-vits16": "facebook/dino-vits16",
    "beit-base": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-large": "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "deit-base": "facebook/deit-base-patch16-224", "deit-small": "facebook/deit-small-patch16-224",
    "vit-base": "google/vit-base-patch16-224-in21k", "vit-large": "google/vit-large-patch16-224-in21k",
    "vit-mae-base": "facebook/vit-mae-base", "vit-mae-large": "facebook/vit-mae-large",
    "vit-msn-base": "facebook/vit-msn-base", "vit-msn-large": "facebook/vit-msn-large",
    "data2vec-base": "facebook/data2vec-vision-base", "data2vec-large": "facebook/data2vec-vision-large",
    "clip-base-32": "openai/clip-vit-base-patch32", "clip-base-16": "openai/clip-vit-base-patch16",
    "clip-large-14": "openai/clip-vit-large-patch14",
    "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
    "swin-small": "microsoft/swin-small-patch4-window7-224",
    "swin-large": "microsoft/swin-large-patch4-window7-224",
    "sam-base": "facebook/sam-vit-base", "sam-large": "facebook/sam-vit-large",
    "sam-huge": "facebook/sam-vit-huge",
}

CNN_REGISTRY = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, ["layer1","layer2","layer3","layer4"]),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, ["layer1","layer2","layer3","layer4"]),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT,
                    ["features.denseblock1","features.denseblock2","features.denseblock3","features.denseblock4"]),
    "densenet201": (models.densenet201, models.DenseNet201_Weights.DEFAULT,
                    ["features.denseblock1","features.denseblock2","features.denseblock3","features.denseblock4"]),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT,
                        ["features.2","features.3","features.5","features.7"]),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT,
                        ["features.2","features.3","features.5","features.7"]),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT,
                      ["features.1","features.3","features.5","features.7"]),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT,
                      ["features.1","features.3","features.5","features.7"]),
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT, ["features.8","features.16","features.23","features.30"]),
    "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT, ["features.9","features.18","features.27","features.36"]),
}

ALL_VISION = list(VIT_MODELS.keys()) + list(CNN_REGISTRY.keys())


def extract_vision_random(images, model_name, device, batch_size=8):
    """从 PIL images 提取 vision embedding，返回 (n_layers, N, d)"""
    is_cnn = model_name in CNN_REGISTRY

    if is_cnn:
        model_fn, weights, target_layers = CNN_REGISTRY[model_name]
        base = model_fn(weights=weights).eval().to(device)
        features = {}
        def get_hook(name):
            def hook(m, i, o): features[name] = o
            return hook
        for name in target_layers:
            parts = name.split(".")
            module = base
            for p in parts: module = getattr(module, p)
            module.register_forward_hook(get_hook(name))
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        all_feats = {k: [] for k in target_layers}
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            imgs = torch.stack([transform(img) for img in batch]).to(device)
            with torch.no_grad():
                _ = base(imgs)
                for k in target_layers:
                    f = features[k]
                    if f.dim() == 4: f = f.mean(dim=[2,3])
                    all_feats[k].append(f.cpu().numpy())
        X_layers = [np.concatenate(all_feats[k], axis=0) for k in target_layers]
        del base, features
    else:
        full_name = VIT_MODELS[model_name]
        try:
            extractor = AutoImageProcessor.from_pretrained(full_name)
        except:
            extractor = AutoFeatureExtractor.from_pretrained(full_name)

        model_lower = full_name.lower()
        if "clip" in model_lower:
            fm = CLIPModel.from_pretrained(full_name, output_hidden_states=True)
            model = fm.vision_model; model.config = fm.config.vision_config
            model.config.model_type = "clip_vision"
        elif "sam" in model_lower:
            fm = SamModel.from_pretrained(full_name, output_hidden_states=True)
            model = fm.vision_encoder; model.config = fm.config.vision_config
            model.config.model_type = "sam"
        else:
            model = AutoModel.from_pretrained(full_name, output_hidden_states=True)
        model = model.to(device).eval()

        model_type = getattr(model.config, "model_type", "")
        n_layers = None
        collectors = {}
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = extractor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                if model_type == "sam":
                    out = model(inputs["pixel_values"], output_hidden_states=True)
                elif model_type == "clip_vision":
                    out = model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
                else:
                    out = model(**inputs)
                hs = out.hidden_states
            has_cls = model_type not in NO_CLS_TYPES
            embeds = []
            for h in hs:
                if has_cls:
                    embeds.append(h[:, 1:, :].mean(dim=1).detach())
                elif h.dim() == 4:
                    embeds.append(h.mean(dim=[1,2]).detach())
                else:
                    embeds.append(h.mean(dim=1).detach())
            if n_layers is None:
                n_layers = len(embeds)
                collectors = {li: [] for li in range(n_layers)}
            for li in range(n_layers):
                collectors[li].append(embeds[li].cpu().numpy())
        X_layers = [np.concatenate(collectors[li], axis=0) for li in range(n_layers)]
        del model

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    X_layers, _ = align_layer_dims(X_layers)
    return np.stack(X_layers, axis=0).astype(np.float16)


# ####################################################################
#  Audio 提取
# ####################################################################

AUDIO_MODELS = {
    "data2vec-audio-base": "facebook/data2vec-audio-base",
    "data2vec-audio-base-960h": "facebook/data2vec-audio-base-960h",
    "data2vec-audio-large": "facebook/data2vec-audio-large",
    "data2vec-audio-large-960h": "facebook/data2vec-audio-large-960h",
    "hubert-base-ls960": "facebook/hubert-base-ls960",
    "hubert-base-superb-ks": "superb/hubert-base-superb-ks",
    "hubert-large-ls960-ft": "facebook/hubert-large-ls960-ft",
    "hubert-xlarge-ls960-ft": "facebook/hubert-xlarge-ls960-ft",
    "sew-d-mid-100k": "asapp/sew-d-mid-100k",
    "sew-d-small-100k": "asapp/sew-d-small-100k",
    "sew-d-tiny-100k": "asapp/sew-d-tiny-100k",
    "sew-mid-100k": "asapp/sew-mid-100k",
    "sew-small-100k": "asapp/sew-small-100k",
    "sew-tiny-100k": "asapp/sew-tiny-100k",
    "unispeech-large-1500h-cv": "microsoft/unispeech-large-1500h-cv",
    "unispeech-sat-base": "microsoft/unispeech-sat-base",
    "unispeech-sat-base-plus": "microsoft/unispeech-sat-base-plus",
    "unispeech-sat-large": "microsoft/unispeech-sat-large",
    "w2v-bert-2.0": "facebook/w2v-bert-2.0",
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-base-960h": "facebook/wav2vec2-base-960h",
    "wav2vec2-base-superb-ks": "superb/wav2vec2-base-superb-ks",
    "wav2vec2-conformer-rel-pos-large": "facebook/wav2vec2-conformer-rel-pos-large",
    "wav2vec2-conformer-rope-large-960h-ft": "facebook/wav2vec2-conformer-rope-large-960h-ft",
    "wav2vec2-large": "facebook/wav2vec2-large",
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "wav2vec2-large-xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    "wav2vec2-xls-r-1b": "facebook/wav2vec2-xls-r-1b",
    "wav2vec2-xls-r-300m": "facebook/wav2vec2-xls-r-300m",
    "wavlm-base": "microsoft/wavlm-base",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    "whisper-base": "openai/whisper-base",
    "whisper-medium": "openai/whisper-medium",
    "whisper-small": "openai/whisper-small",
    "whisper-tiny": "openai/whisper-tiny",
}

ALL_AUDIO = list(AUDIO_MODELS.keys())


def extract_audio_random(audios, model_name, device, sr=16000):
    """对随机音频提取 embedding，每条音频一个 (n_layers, 1, d)"""
    full_name = AUDIO_MODELS[model_name]
    model_lower = full_name.lower()

    if "whisper" in model_lower:
        processor = WhisperFeatureExtractor.from_pretrained(full_name)
        fm = WhisperModel.from_pretrained(full_name, output_hidden_states=True)
        model = fm.encoder; model.config = fm.config; model.config.model_type = "whisper"
    else:
        try:
            processor = AutoProcessor.from_pretrained(full_name)
        except:
            from transformers import AutoFeatureExtractor as AFE
            processor = AFE.from_pretrained(full_name)
        model = AutoModel.from_pretrained(full_name, output_hidden_states=True)

    model = model.to(device).eval()
    model_type = getattr(model.config, "model_type", "")

    all_embeddings = []  # list of (n_layers, d)

    for audio in tqdm(audios, desc=f"  {model_name}", ncols=80, leave=False):
        if model_type == "whisper":
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(inputs["input_features"], output_hidden_states=True)
                hs = out.hidden_states
        else:
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states

        layers = []
        for h in hs:
            layers.append(h.mean(dim=1).squeeze(0).cpu().numpy())
        all_embeddings.append(np.stack(layers, axis=0))  # (n_layers, d)

    del model, processor
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # (n_layers, N, d)
    X = np.stack(all_embeddings, axis=1).astype(np.float16)
    return X


# ####################################################################
#  Language 提取
# ####################################################################

LANG_MODELS = {
    "MiniLM-L6-H384-uncased": "nreimers/MiniLM-L6-H384-uncased",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "mpnet-base": "microsoft/mpnet-base",
    "albert-base-v2": "albert-base-v2", "albert-large-v2": "albert-large-v2",
    "albert-xlarge-v2": "albert-xlarge-v2",
    "bert-base-cased": "bert-base-cased", "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "bert-base-uncased": "bert-base-uncased", "bert-large-cased": "bert-large-cased",
    "bert-large-uncased": "bert-large-uncased",
    "camembert-base": "camembert-base",
    "conv-bert-base": "YituTech/conv-bert-base", "conv-bert-medium-small": "YituTech/conv-bert-medium-small",
    "data2vec-text-base": "facebook/data2vec-text-base",
    "deberta-base": "microsoft/deberta-base", "deberta-large": "microsoft/deberta-large",
    "distilbert-base-multilingual-cased": "distilbert-base-multilingual-cased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilroberta-base": "distilroberta-base",
    "electra-base-discriminator": "google/electra-base-discriminator",
    "electra-large-discriminator": "google/electra-large-discriminator",
    "electra-small-discriminator": "google/electra-small-discriminator",
    "ernie-2.0-base-en": "nghuyong/ernie-2.0-base-en", "ernie-2.0-large-en": "nghuyong/ernie-2.0-large-en",
    "ibert-roberta-base": "kssteven/ibert-roberta-base",
    "rembert": "google/rembert",
    "roberta-base": "roberta-base", "roberta-large": "roberta-large",
    "squeezebert-uncased": "squeezebert/squeezebert-uncased",
    "t5-small": "google-t5/t5-small",
    "xlm-roberta-base": "xlm-roberta-base", "xlm-roberta-large": "xlm-roberta-large",
    "xlnet-base-cased": "xlnet-base-cased", "xlnet-large-cased": "xlnet-large-cased",
}

ALL_LANG = list(LANG_MODELS.keys())


def extract_language_random(words, model_name, device, context_window=32, words_per_bin=5):
    """对随机词序列提取 embedding，返回 (n_layers, n_bins, d)"""
    full_name = LANG_MODELS[model_name]
    model_lower = full_name.lower()

    needs_prefix_space = any(k in model_lower for k in ["gpt2", "roberta", "bart"])
    tokenizer_kwargs = {"use_fast": True}
    if needs_prefix_space:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(full_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_lower:
        model = T5EncoderModel.from_pretrained(full_name, output_hidden_states=True)
        model.config.model_type = "t5_encoder"
    else:
        model = AutoModel.from_pretrained(full_name, output_hidden_states=True)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device).eval()

    # 词级编码
    all_layers = None
    for i in tqdm(range(len(words)), desc=f"  {model_name}", ncols=80, leave=False):
        start = max(0, i - context_window)
        context_words = [str(w) for w in words[start:i + 1]]
        target_idx = len(context_words) - 1

        enc = tokenizer(context_words, is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=512, padding=False).to(device)
        word_ids = enc.word_ids(batch_index=0)
        target_positions = [t for t, w in enumerate(word_ids) if w == target_idx]

        if not target_positions:
            enc = tokenizer([str(words[i])], is_split_into_words=True, return_tensors="pt",
                            truncation=True, max_length=512, padding=False).to(device)
            word_ids = enc.word_ids(batch_index=0)
            target_positions = [t for t, w in enumerate(word_ids) if w == 0]

        with torch.no_grad():
            out = model(**enc)
            hs = out.hidden_states

        pos_t = torch.tensor(target_positions, device=device)
        vecs = []
        for h in hs:
            vecs.append(h[0, pos_t, :].mean(dim=0).cpu().numpy())
        vecs = np.stack(vecs, axis=0)

        if all_layers is None:
            n_layers, feat_dim = vecs.shape
            all_layers = np.zeros((n_layers, len(words), feat_dim), dtype=np.float32)
        all_layers[:, i, :] = vecs

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 分箱
    n_layers, n_words, feat_dim = all_layers.shape
    n_bins = int(np.ceil(n_words / words_per_bin))
    X_binned = np.zeros((n_layers, n_bins, feat_dim), dtype=np.float32)
    for b in range(n_bins):
        s = b * words_per_bin
        e = min(s + words_per_bin, n_words)
        X_binned[:, b, :] = all_layers[:, s:e, :].mean(axis=1)

    return X_binned


# ####################################################################
#  Eigvals 提取
# ####################################################################

def compute_eigvals_from_embedding(X):
    """X: (n_layers, N, d) → eigvals array"""
    if X.ndim != 3:
        return np.array([])
    L, N, d = X.shape
    all_eigs = []
    for t in range(N):
        eigvals = compute_dmd_eigenvalues(X[:, t, :])
        if eigvals is not None:
            all_eigs.extend(eigvals)
    return np.array(all_eigs)


# ####################################################################
#  主流程
# ####################################################################

def run_vision(device, batch_size=8, n_samples=1000):
    print("\n" + "="*60)
    print("  VISION - Random Gaussian Images")
    print("="*60)

    save_emb = "embeddings/random_vision"
    save_eig = "neweigvals/random_vision"
    os.makedirs(save_emb, exist_ok=True)
    os.makedirs(save_eig, exist_ok=True)

    images = generate_random_images(n=n_samples)
    print(f"  生成 {len(images)} 张随机图片")

    for model_name in ALL_VISION:
        emb_path = os.path.join(save_emb, f"{model_name}.npy")
        eig_path = os.path.join(save_eig, f"{model_name}.npy")

        if os.path.exists(eig_path):
            print(f"  已存在，跳过: {model_name}")
            continue

        print(f"\n  {model_name}")
        try:
            X = extract_vision_random(images, model_name, device, batch_size)
            np.save(emb_path, X)
            print(f"    embedding: {X.shape}")

            eigvals = compute_eigvals_from_embedding(X.astype(np.float32))
            np.save(eig_path, eigvals)
            print(f"    eigvals: {eigvals.shape}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_audio(device, n_samples=1000):
    print("\n" + "="*60)
    print("  AUDIO - Random Gaussian Noise")
    print("="*60)

    save_emb = "embeddings/random_audio"
    save_eig = "neweigvals/random_audio"
    os.makedirs(save_emb, exist_ok=True)
    os.makedirs(save_eig, exist_ok=True)

    audios = generate_random_audio(n=n_samples)
    print(f"  生成 {len(audios)} 条随机音频")

    for model_name in ALL_AUDIO:
        eig_path = os.path.join(save_eig, f"{model_name}.npy")

        if os.path.exists(eig_path):
            print(f"  已存在，跳过: {model_name}")
            continue

        print(f"\n  {model_name}")
        try:
            X = extract_audio_random(audios, model_name, device)

            # 保存 embedding
            model_emb_dir = os.path.join(save_emb, model_name)
            os.makedirs(model_emb_dir, exist_ok=True)
            np.save(os.path.join(model_emb_dir, "random.npy"), X)
            print(f"    embedding: {X.shape}")

            eigvals = compute_eigvals_from_embedding(X.astype(np.float32))
            np.save(eig_path, eigvals)
            print(f"    eigvals: {eigvals.shape}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_language(device, n_samples=1000):
    print("\n" + "="*60)
    print("  LANGUAGE - Random Words")
    print("="*60)

    save_emb = "embeddings/random_language"
    save_eig = "neweigvals/random_language"
    os.makedirs(save_emb, exist_ok=True)
    os.makedirs(save_eig, exist_ok=True)

    for model_name in ALL_LANG:
        eig_path = os.path.join(save_eig, f"{model_name}.npy")

        if os.path.exists(eig_path):
            print(f"  已存在，跳过: {model_name}")
            continue

        print(f"\n  {model_name}")
        try:
            # 每个模型用自己的 tokenizer 生成随机词
            full_name = LANG_MODELS[model_name]
            model_lower = full_name.lower()
            needs_prefix = any(k in model_lower for k in ["gpt2", "roberta", "bart"])
            tk_kwargs = {"use_fast": True}
            if needs_prefix: tk_kwargs["add_prefix_space"] = True
            tokenizer = AutoTokenizer.from_pretrained(full_name, **tk_kwargs)

            words = generate_random_words(tokenizer, n=n_samples)
            del tokenizer

            X = extract_language_random(words, model_name, device)

            model_emb_dir = os.path.join(save_emb, model_name)
            os.makedirs(model_emb_dir, exist_ok=True)
            np.save(os.path.join(model_emb_dir, "random.npy"), X)
            print(f"    embedding: {X.shape}")

            eigvals = compute_eigvals_from_embedding(X.astype(np.float32))
            np.save(eig_path, eigvals)
            print(f"    eigvals: {eigvals.shape}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


# ####################################################################
#  Spectral Analysis (复用)
# ####################################################################

def run_spectral_analysis():
    """复用 spectral analysis 逻辑，对比 random vs real"""
    from scipy.stats import gaussian_kde, mannwhitneyu

    print("\n" + "="*60)
    print("  SPECTRAL ANALYSIS: Random vs Real")
    print("="*60)

    modalities = ["vision", "audio", "language"]

    for mod in modalities:
        real_dir = f"neweigvals/{mod}"
        rand_dir = f"neweigvals/random_{mod}"

        if not os.path.exists(real_dir) or not os.path.exists(rand_dir):
            print(f"  ⚠️ 跳过 {mod}: 缺少数据")
            continue

        real_means = []
        rand_means = []

        real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.npy')])
        rand_files = sorted([f for f in os.listdir(rand_dir) if f.endswith('.npy')])

        for f in real_files:
            eigvals = np.load(os.path.join(real_dir, f), allow_pickle=True).flatten()
            eigvals_abs = np.abs(eigvals)
            if len(eigvals_abs) > 0 and np.all(np.isfinite(eigvals_abs)):
                real_means.append(np.mean(eigvals_abs))

        for f in rand_files:
            eigvals = np.load(os.path.join(rand_dir, f), allow_pickle=True).flatten()
            eigvals_abs = np.abs(eigvals)
            if len(eigvals_abs) > 0 and np.all(np.isfinite(eigvals_abs)):
                rand_means.append(np.mean(eigvals_abs))

        if not real_means or not rand_means:
            print(f"  ⚠️ {mod}: 数据不足")
            continue

        real_means = np.array(real_means)
        rand_means = np.array(rand_means)

        u_stat, p_val = mannwhitneyu(real_means, rand_means, alternative='two-sided')

        print(f"\n  {mod.upper()}:")
        print(f"    Real  mean|λ|: {np.mean(real_means):.4f} ± {np.std(real_means):.4f} (n={len(real_means)})")
        print(f"    Random mean|λ|: {np.mean(rand_means):.4f} ± {np.std(rand_means):.4f} (n={len(rand_means)})")
        print(f"    Mann-Whitney U={u_stat:.1f}, p={p_val:.4e}")
        if p_val < 0.05:
            direction = "Real > Random" if np.mean(real_means) > np.mean(rand_means) else "Real < Random"
            print(f"    → 显著差异: {direction}")
        else:
            print(f"    → 无显著差异")


# ####################################################################
#  CLI
# ####################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Random Baseline Experiment")
    parser.add_argument("--step", type=str, default="all",
                        choices=["extract", "analysis", "all"])
    parser.add_argument("--modality", type=str, default="all",
                        choices=["vision", "audio", "language", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = ["vision", "audio", "language"] if args.modality == "all" else [args.modality]

    if args.step in ("extract", "all"):
        if "vision" in modalities:
            run_vision(args.device, args.batch_size, args.n_samples)
        if "audio" in modalities:
            run_audio(args.device, args.n_samples)
        if "language" in modalities:
            run_language(args.device, args.n_samples)

    if args.step in ("analysis", "all"):
        run_spectral_analysis()

    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()