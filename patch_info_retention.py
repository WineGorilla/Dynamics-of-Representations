"""
Per-Patch Information Retention Across Dynamical Modes
======================================================
对每个模型的每张图：
  1. 提取所有层的 per-patch token: (L, N_patches, D)
  2. 对每个 patch 位置独立做 fuse_layers_single_soft_dmd (各 center)
  3. cos(fused_patch, last_layer_patch) = 信息保留度
  4. Random baseline: 随机选一半模式融合
  5. 统计检验 + 跨模型汇总

支持：ViT 系列 (DINO, DINOv2, BEiT, MAE, MSN, CLIP, Swin, DeiT, Data2Vec)
      CNN 系列 (ResNet, DenseNet, EfficientNet, ConvNeXt, VGG)

用法：
  CUDA_VISIBLE_DEVICES=1 python patch_info_retention.py --device cuda
  CUDA_VISIBLE_DEVICES=1 python patch_info_retention.py --device cuda --max_img 50  # 快速测试
  CUDA_VISIBLE_DEVICES=1 python patch_info_retention.py --device cuda --model_type cnn
"""

import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from scipy.stats import friedmanchisquare
from PIL import Image
from transformers import AutoImageProcessor, AutoFeatureExtractor, AutoModel, CLIPModel, SamModel
from torchvision import models, transforms

from core.dmd import fuse_layers_single_soft_dmd


# ═══════════════════════════════════════════════════════════════
#  ViT 模型加载 + per-patch 提取
# ═══════════════════════════════════════════════════════════════

NO_CLS_TYPES = {"swin", "sam", "siglip"}


def load_vit_model(model_name, device):
    """加载 ViT 系列模型，返回 (processor, model)"""
    model_lower = model_name.lower()

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_name)

    if "clip" in model_lower or "siglip" in model_lower:
        full_model = CLIPModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.vision_model
        model.config = full_model.config.vision_config
        model.config.model_type = "clip_vision"
    elif "sam" in model_lower:
        full_model = SamModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.vision_encoder
        model.config = full_model.config.vision_config
        model.config.model_type = "sam"
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device).eval()
    return processor, model


def extract_vit_patch_tokens(processor, model, img, device):
    """
    单张图 → (L, N_patches, D)
    根据模型类型处理 CLS token 和 4D hidden states。
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    model_type = getattr(model.config, "model_type", "")

    with torch.no_grad():
        if model_type == "clip_vision":
            outputs = model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        elif model_type == "sam":
            outputs = model(inputs["pixel_values"], output_hidden_states=True)
        else:
            outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    has_cls = model_type not in NO_CLS_TYPES

    patch_tokens = []
    for h in hidden_states:
        if h.dim() == 4:
            # (B, H, W, D) → (H*W, D)
            x = h[0].reshape(-1, h.shape[-1]).cpu().numpy()
        elif has_cls:
            # (B, 1+N, D) → (N, D)
            x = h[0, 1:, :].cpu().numpy()
        else:
            # (B, N, D) → (N, D)
            x = h[0].cpu().numpy()
        patch_tokens.append(x)

    return np.stack(patch_tokens, axis=0).astype(np.float32)  # (L, N_patches, D)


# ═══════════════════════════════════════════════════════════════
#  CNN 模型加载 + per-patch 提取（保留空间维度）
# ═══════════════════════════════════════════════════════════════

CNN_CONFIGS = {
    "resnet18":        (models.resnet18,        models.ResNet18_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "resnet34":        (models.resnet34,        models.ResNet34_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "resnet101":       (models.resnet101,       models.ResNet101_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "densenet121":     (models.densenet121,      models.DenseNet121_Weights.DEFAULT,
                        ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]),
    "densenet201":     (models.densenet201,      models.DenseNet201_Weights.DEFAULT,
                        ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]),
    "efficientnet_b0": (models.efficientnet_b0,  models.EfficientNet_B0_Weights.DEFAULT,
                        ["features.2", "features.3", "features.5", "features.7"]),
    "efficientnet_b4": (models.efficientnet_b4,  models.EfficientNet_B4_Weights.DEFAULT,
                        ["features.2", "features.3", "features.5", "features.7"]),
    "convnext_tiny":   (models.convnext_tiny,    models.ConvNeXt_Tiny_Weights.DEFAULT,
                        ["features.1", "features.3", "features.5", "features.7"]),
    "convnext_base":   (models.convnext_base,    models.ConvNeXt_Base_Weights.DEFAULT,
                        ["features.1", "features.3", "features.5", "features.7"]),
    "vgg16":           (models.vgg16,            models.VGG16_Weights.DEFAULT,
                        ["features.8", "features.16", "features.23", "features.30"]),
    "vgg19":           (models.vgg19,            models.VGG19_Weights.DEFAULT,
                        ["features.9", "features.18", "features.27", "features.36"]),
}


def load_cnn_model(model_name, device):
    """加载 CNN 模型，注册 hook 捕获中间层特征图。返回 (model, transform, target_layers, features_dict)"""
    if model_name not in CNN_CONFIGS:
        raise ValueError(f"Unsupported CNN: {model_name}")

    factory, weights, target_layers = CNN_CONFIGS[model_name]
    base = factory(weights=weights).eval().to(device)

    features = {}

    def get_hook(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    for name in target_layers:
        parts = name.split(".")
        module = base
        for p in parts:
            module = getattr(module, p)
        module.register_forward_hook(get_hook(name))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return base, transform, target_layers, features


def random_project_spatial(feat_map, target_dim, rng):
    """
    将空间特征图的通道维度投影到 target_dim。
    feat_map: (H, W, C) → (H, W, target_dim)
    """
    C = feat_map.shape[-1]
    if C == target_dim:
        return feat_map
    R = rng.randn(C, target_dim).astype(np.float32) / np.sqrt(target_dim)
    H, W, _ = feat_map.shape
    return (feat_map.reshape(-1, C) @ R).reshape(H, W, target_dim)


def extract_cnn_patch_tokens(cnn_model, transform, target_layers, features, img, device):
    """
    单张图 → (L, N_spatial, D)
    CNN 各层特征图空间位置不同，统一到最小空间分辨率 + 统一通道维度。
    """
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = cnn_model(x)

    # 收集各层特征图: (C, H, W) → (H, W, C)
    feat_maps = []
    for name in target_layers:
        fm = features[name][0].cpu().numpy()  # (C, H, W)
        fm = fm.transpose(1, 2, 0)  # (H, W, C)
        feat_maps.append(fm)

    # 统一空间分辨率：下采样到最小 H, W
    min_h = min(fm.shape[0] for fm in feat_maps)
    min_w = min(fm.shape[1] for fm in feat_maps)

    # 简单方式：自适应平均池化到 (min_h, min_w)
    resized = []
    for fm in feat_maps:
        if fm.shape[0] != min_h or fm.shape[1] != min_w:
            # 用 torch 做自适应池化
            t = torch.from_numpy(fm.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, C, H, W)
            t = torch.nn.functional.adaptive_avg_pool2d(t, (min_h, min_w))
            fm = t[0].numpy().transpose(1, 2, 0)  # (min_h, min_w, C)
        resized.append(fm)

    # 统一通道维度：随机投影到最小通道数
    channel_dims = [fm.shape[-1] for fm in resized]
    target_dim = min(channel_dims)
    rng = np.random.RandomState(42)

    unified = []
    for fm in resized:
        if fm.shape[-1] != target_dim:
            fm = random_project_spatial(fm, target_dim, rng)
        unified.append(fm.reshape(-1, target_dim))  # (H*W, D)

    return np.stack(unified, axis=0).astype(np.float32)  # (L, N_spatial, D)


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def format_p(p):
    if p < 0.0001:   return "< 0.0001"
    elif p < 0.001:  return "< 0.001"
    elif p < 0.01:   return "< 0.01"
    elif p < 0.05:   return "< 0.05"
    else:            return f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════
#  核心分析：处理一张图的所有 patch
# ═══════════════════════════════════════════════════════════════

def process_patches(patch_tokens, centers, dmd_k=3, sigma=0.1):
    """
    patch_tokens: (L, N_patches, D)
    对每个 patch 做 DMD fusion，返回 per-patch similarity。
    """
    L, N_patches, D = patch_tokens.shape
    last_layer = patch_tokens[-1]  # (N_patches, D)

    sims = {c: [] for c in centers}

    for p in range(N_patches):
        trajectory = patch_tokens[:, p, :]  # (L, D)

        for c in centers:
            try:
                fused = fuse_layers_single_soft_dmd(trajectory, r=dmd_k, center=c, sigma=sigma)
                sims[c].append(cosine_sim(fused, last_layer[p]))
            except Exception:
                sims[c].append(0.0)

    return sims


# ═══════════════════════════════════════════════════════════════
#  单个模型
# ═══════════════════════════════════════════════════════════════

def run_one_model(model_name, img_paths, centers, device="cuda",
                  sigma=0.1, dmd_k=3, model_type="vit"):
    """统一入口：ViT 和 CNN 共用后续分析逻辑。"""

    # 加载模型
    if model_type == "vit":
        processor, model = load_vit_model(model_name, device)
        extract_fn = lambda img: extract_vit_patch_tokens(processor, model, img, device)
        tag = model_name.split("/")[-1]
    else:
        cnn_model, transform, target_layers, features = load_cnn_model(model_name, device)
        extract_fn = lambda img: extract_cnn_patch_tokens(cnn_model, transform, target_layers, features, img, device)
        tag = model_name

    # 收集所有 patch 的 similarity
    all_sims = {c: [] for c in centers}

    for img_path in tqdm(img_paths, desc=f"  {tag}"):
        try:
            img = Image.open(img_path).convert("RGB")
            patch_tokens = extract_fn(img)
        except Exception as e:
            continue

        sims = process_patches(patch_tokens, centers, dmd_k, sigma)

        for c in centers:
            all_sims[c].extend(sims[c])

    n_total = len(all_sims[centers[0]])
    if n_total < 100:
        print(f"    ⚠️ {tag}: 数据不足 ({n_total})")
        if model_type == "vit":
            del processor, model
        else:
            del cnn_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # ── 统计检验：Friedman test (三个 center 之间是否有差异) ──
    from scipy.stats import friedmanchisquare

    center_means = {c: float(np.mean(all_sims[c])) for c in centers}
    best_center = max(center_means, key=center_means.get)

    # Friedman 需要配对数据，采样到可控大小
    max_n = 50000
    n_use = min(n_total, max_n)
    if n_total > max_n:
        idx = np.random.choice(n_total, size=max_n, replace=False)
    else:
        idx = np.arange(n_total)

    samples = [np.array(all_sims[c])[idx] for c in centers]
    try:
        chi2, p_friedman = friedmanchisquare(*samples)
    except Exception:
        chi2, p_friedman = 0.0, 1.0

    def _sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    cm_str = " | ".join([f"c={c}:{center_means[c]:.4f}" for c in centers])
    print(f"    {tag:35s} | n={n_total:>6} | {cm_str} | best=c={best_center} | "
          f"χ²={chi2:.1f} p={format_p(p_friedman)}{_sig(p_friedman)}")

    # 释放模型
    if model_type == "vit":
        del processor, model
    else:
        del cnn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_type": model_type,
        "n_patches_total": n_total,
        "center_means": center_means,
        "best_center": best_center,
        "friedman_chi2": chi2,
        "friedman_p": p_friedman,
        "sims": {c: np.array(all_sims[c]) for c in centers},
    }


# ═══════════════════════════════════════════════════════════════
#  模态级汇总
# ═══════════════════════════════════════════════════════════════

def aggregate_results(model_results, group_name, centers):
    """
    模态级汇总：把每个模型的 center_means 作为一个观测，
    对所有模型做 Friedman test，检验 center 之间是否存在跨模型的系统性差异。
    """
    n = len(model_results)
    n_sig = sum(1 for r in model_results if r["friedman_p"] < 0.05)

    # 每个 center 在所有模型上的均值向量
    # shape: (n_models,) per center
    center_vectors = {}
    for c in centers:
        center_vectors[c] = np.array([r["center_means"][c] for r in model_results])

    # 跨模型 Friedman test：以模型为 block，center 为 treatment
    samples = [center_vectors[c] for c in centers]
    if n >= 3:
        try:
            chi2_global, p_global = friedmanchisquare(*samples)
        except Exception:
            chi2_global, p_global = 0.0, 1.0
    else:
        chi2_global, p_global = 0.0, 1.0

    # 每个 center 的跨模型均值 ± std
    center_summary = {}
    for c in centers:
        vals = center_vectors[c]
        center_summary[c] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0}

    return {
        "group": group_name,
        "n_models": n,
        "n_significant": n_sig,
        "center_summary": center_summary,
        "friedman_chi2": chi2_global,
        "friedman_p": p_global,
        "model_results": model_results,
    }


# ═══════════════════════════════════════════════════════════════
#  表格
# ═══════════════════════════════════════════════════════════════

def print_table1(group_summaries, centers):
    print("\n" + "=" * 150)
    print("Table 1: Per-Patch Information Retention — cos(DMD_fused, last_layer) by Center")
    print("  Higher = DMD mode captures more of the final representation at each position")
    print("=" * 150)

    for gs in group_summaries:
        print(f"\n── {gs['group']} ({gs['n_models']} models, {gs['n_significant']}/{gs['n_models']} significant) ──")

        c_hdrs = " | ".join([f"c={c:>4}" for c in centers])
        header = f"  {'Model':<35} | {'N':>8} | {c_hdrs} | {'Best':>5} | {'χ²':>8} | {'p':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for r in gs["model_results"]:
            cm = r["center_means"]
            c_vals = " | ".join([f"{cm[c]:>6.4f}" for c in centers])
            name = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
            def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
            print(f"  {name:<35} | {r['n_patches_total']:>8} | {c_vals} | c={r['best_center']:<3} | "
                  f"{r['friedman_chi2']:>8.1f} | {format_p(r['friedman_p']):>10} {_s(r['friedman_p'])}")

    print("=" * 150)


def print_table2(group_summaries, centers):
    """Table 2: Friedman omnibus + Wilcoxon post-hoc on model-level means"""
    from scipy.stats import wilcoxon as wilcoxon_test

    print("\n" + "=" * 115)
    print("Table 2: Cross-Model Universality")
    print("  Each model contributes one observation (its mean similarity per center).")
    print("  Friedman omnibus test, followed by Wilcoxon signed-rank post-hoc (two-sided).")
    print("=" * 115)

    def _s(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

    for gs in group_summaries:
        n = gs["n_models"]

        # Friedman omnibus
        print(f"\n  {gs['group']}  (n = {n} models)")
        print(f"  Friedman χ²({len(centers)-1}) = {gs['friedman_chi2']:.2f},  p = {format_p(gs['friedman_p'])} {_s(gs['friedman_p'])}")
        print()

        # Post-hoc
        header = f"  {'Test':<8} | {'#Models':>7} | {'Mean Δ':>8} | {'Cohen d':>8} | {'W-stat':>8} | {'p-value':>12} | {'Sig':>5}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        vals = {c: np.array([r["center_means"][c] for r in gs["model_results"]]) for c in centers}

        pairs = [
            ("S > T", centers[2], centers[0]),
            ("S > M", centers[2], centers[1]),
            ("M > T", centers[1], centers[0]),
        ]

        for label, c_a, c_b in pairs:
            diff = vals[c_a] - vals[c_b]
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1)) if n > 1 else 1.0
            cohens_d = mean_d / std_d if std_d > 0 else float('inf')

            if n >= 2 and not np.all(diff == 0):
                w, p = wilcoxon_test(diff, alternative='two-sided')
            else:
                w, p = 0.0, 1.0

            d_str = f"{cohens_d:.2f}" if not np.isinf(cohens_d) else "inf"
            print(f"  {label:<8} | {n:>7} | {mean_d:>+8.4f} | {d_str:>8} | {w:>8.1f} | {format_p(p):>12} | {_s(p):>5}")

        print(f"  {'-' * (len(header) - 2)}")

    print("=" * 115)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

VISION_MODELS = [
    # ── ViT 系列 ──
    # BEiT
    ("microsoft/beit-base-patch16-224-pt22k-ft22k", "vit"),
    ("microsoft/beit-large-patch16-224-pt22k-ft22k", "vit"),
    # CLIP
    ("openai/clip-vit-base-patch16", "vit"),
    ("openai/clip-vit-base-patch32", "vit"),
    ("openai/clip-vit-large-patch14", "vit"),
    # Data2Vec
    ("facebook/data2vec-vision-base", "vit"),
    ("facebook/data2vec-vision-large", "vit"),
    # DeiT
    ("facebook/deit-base-patch16-224", "vit"),
    ("facebook/deit-small-patch16-224", "vit"),
    # DINO
    ("facebook/dino-vitb16", "vit"),
    ("facebook/dino-vits16", "vit"),
    # DINOv2
    ("facebook/dinov2-base", "vit"),
    ("facebook/dinov2-large", "vit"),
    ("facebook/dinov2-small", "vit"),
    # SAM
    ("facebook/sam-vit-base", "vit"),
    ("facebook/sam-vit-large", "vit"),
    ("facebook/sam-vit-huge", "vit"),
    # Swin
    ("microsoft/swin-tiny-patch4-window7-224", "vit"),
    ("microsoft/swin-small-patch4-window7-224", "vit"),
    ("microsoft/swin-base-patch4-window7-224", "vit"),
    ("microsoft/swin-large-patch4-window7-224", "vit"),
    # ViT
    ("google/vit-base-patch16-224-in21k", "vit"),
    ("google/vit-large-patch16-224-in21k", "vit"),
    # MAE
    ("facebook/vit-mae-base", "vit"),
    ("facebook/vit-mae-large", "vit"),
    # MSN
    ("facebook/vit-msn-base", "vit"),
    ("facebook/vit-msn-large", "vit"),
    # ── CNN 系列 ──
    ("resnet18", "cnn"),
    ("resnet34", "cnn"),
    ("resnet50", "cnn"),
    ("resnet101", "cnn"),
    ("wide_resnet50_2", "cnn"),
    ("densenet121", "cnn"),
    ("densenet201", "cnn"),
    ("efficientnet_b0", "cnn"),
    ("efficientnet_b4", "cnn"),
    ("convnext_tiny", "cnn"),
    ("convnext_base", "cnn"),
    ("vgg16", "cnn"),
    ("vgg19", "cnn"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--img_root",   type=str,   default="data/img_data/images")
    parser.add_argument("--sigma",      type=float, default=0.1)
    parser.add_argument("--dmd_k",      type=int,   default=3)
    parser.add_argument("--max_img",    type=int,   default=200)
    parser.add_argument("--save_root",  type=str,   default="results/patch_retention")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    centers = [0.0, 0.5, 1.0]

    # 收集图像
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG"):
        img_paths.extend(glob(os.path.join(args.img_root, "**", ext), recursive=True))
    img_paths = sorted(set(img_paths))[:args.max_img]
    print(f"Images: {len(img_paths)}")
    print(f"Centers = {centers}, sigma = {args.sigma}, dmd_k = {args.dmd_k}")

    all_results = []
    for model_name, model_type in VISION_MODELS:
        print(f"\n  [{model_name}]")
        try:
            r = run_one_model(model_name, img_paths, centers, device=args.device,
                              sigma=args.sigma, dmd_k=args.dmd_k, model_type=model_type)
            if r:
                all_results.append(r)
        except Exception as e:
            print(f"    ⚠️ {model_name}: {e}")

    if not all_results:
        print("⚠️ 没有结果")
        return

    gs = aggregate_results(all_results, "Vision", centers)
    print_table1([gs], centers)
    print_table2([gs], centers)

    # 保存
    save_data = {}
    for r in all_results:
        mkey = r["model"].replace("/", "_")
        for c in centers:
            save_data[f"{mkey}_c{c}"] = r["sims"][c]

    npz_path = os.path.join(args.save_root, "patch_retention_results.npz")
    np.savez(npz_path, **save_data)
    print(f"\n保存: {npz_path}")


if __name__ == "__main__":
    main()