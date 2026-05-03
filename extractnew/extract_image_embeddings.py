"""
统一图像 Embedding 提取脚本（纯图像数据集版）
==============================================
支持模型类型:
  - ViT 系列 (DINOv2, BEiT, DeiT, DINO, ViT-MAE, ViT-MSN, ...)
  - CLIP / SigLIP 视觉编码器
  - SAM 视觉编码器
  - Swin Transformer
  - CNN 系列 (ResNet, DenseNet, EfficientNet, ConvNeXt, VGG)

数据集格式（自动检测）:
  方式1 - ImageNet 风格（按类别分子文件夹）:
      dataset_root/
        class_a/img001.jpg
        class_b/img002.jpg
  方式2 - 平铺所有图片在一个文件夹:
      dataset_root/img001.jpg, img002.jpg, ...

输出格式:
  每个模型保存一个 .npy 文件
  shape = (n_layers, n_images, feat_dim)
  与 fMRI 版本的 (n_layers, n_tr, feat_dim) 保持一致

用法:
python extractnew/extract_image_embeddings.py \
    --data_root /data/mi2-interns/ruiyu/Dynamics-of-Representations/256_ObjectCategories \
    --save_root embeddings \
    --device cuda \
    --batch_size 8
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
import gc
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from PIL import Image

# ─── ViT 相关 ────────────────────────────────────────────
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoImageProcessor,
    CLIPModel,
    SamModel,
)

# ─── CNN 相关 ────────────────────────────────────────────
from torchvision import models, transforms


# ====================================================================
#  图片收集
# ====================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_image_paths(data_root):
    """
    自动检测数据集结构，收集所有图片路径。
    支持 ImageNet 风格子文件夹 和 平铺文件夹。
    返回排序后的路径列表。
    """
    paths = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    print(f"共找到 {len(paths)} 张图片 (根目录: {data_root})")
    return paths


# ====================================================================
#  随机投影降维（保持层间维度一致）
# ====================================================================
def random_project(X, target_dim, random_state=42):
    d = X.shape[1]
    if d == target_dim:
        return X.astype(np.float32)
    rng = np.random.RandomState(random_state)
    R = rng.randn(d, target_dim).astype(np.float32) / np.sqrt(target_dim)
    return X.astype(np.float32) @ R


def align_layer_dims(X_layers):
    """将各层 embedding 统一到最小维度"""
    feat_dims = [X.shape[1] for X in X_layers]
    target_dim = min(feat_dims)
    if len(set(feat_dims)) > 1:
        print(f"  层维度不一致 {feat_dims} → 随机投影到 {target_dim}")
        X_layers = [random_project(X, target_dim) for X in X_layers]
    return X_layers, target_dim


# ####################################################################
#  ViT / Transformer 模型
# ####################################################################

NO_CLS_TYPES = {"swin", "sam", "siglip"}


def load_vit_model(model_name, device="cuda"):
    print(f"加载 ViT 模型: {model_name}")
    try:
        extractor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)

    model_lower = model_name.lower()

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
    return extractor, model


def _vit_forward(model, inputs):
    model_type = getattr(model.config, "model_type", "")
    if model_type == "sam":
        return model(inputs["pixel_values"], output_hidden_states=True)
    elif model_type == "clip_vision":
        return model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
    else:
        return model(**inputs)


def _vit_pool(hidden_states, model, cls_only=False):
    model_type = getattr(model.config, "model_type", "")
    has_cls = model_type not in NO_CLS_TYPES

    if cls_only:
        if not has_cls:
            return [h.mean(dim=1).detach() for h in hidden_states]
        return [h[:, 0, :].detach() for h in hidden_states]
    else:
        if has_cls:
            return [h[:, 1:, :].mean(dim=1).detach() for h in hidden_states]
        else:
            result = []
            for h in hidden_states:
                if h.dim() == 4:
                    result.append(h.mean(dim=[1, 2]).detach())
                else:
                    result.append(h.mean(dim=1).detach())
            return result


def extract_vit_embeddings(extractor, model, image_paths, device, cls_only=False, batch_size=4):
    """提取 ViT 模型各层 embedding，返回 list of (N, d) arrays"""
    n_layers = None
    layer_collectors = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  ViT batches", leave=False):
        batch_paths = image_paths[i : i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  跳过损坏图像: {p} ({e})")
                continue
        if not batch_imgs:
            continue

        try:
            inputs = extractor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = _vit_forward(model, inputs)
            layer_embeds = _vit_pool(outputs.hidden_states, model, cls_only)
        except Exception as e:
            print(f"  Batch 失败，逐张重试: {e}")
            layer_embeds_list = []
            for img in batch_imgs:
                try:
                    inputs = extractor(images=[img], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = _vit_forward(model, inputs)
                    layer_embeds_list.append(_vit_pool(outputs.hidden_states, model, cls_only))
                except Exception as e2:
                    print(f"  跳过问题图像: {e2}")
            if not layer_embeds_list:
                continue
            n_l = len(layer_embeds_list[0])
            layer_embeds = [
                torch.cat([le[li] for le in layer_embeds_list], dim=0) for li in range(n_l)
            ]

        if n_layers is None:
            n_layers = len(layer_embeds)
            layer_collectors = {li: [] for li in range(n_layers)}

        for li in range(n_layers):
            layer_collectors[li].append(layer_embeds[li].cpu().numpy())

    if n_layers is None:
        return []
    return [np.concatenate(layer_collectors[li], axis=0) for li in range(n_layers)]


# ####################################################################
#  CNN 模型
# ####################################################################

CNN_REGISTRY = {
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "resnet101":       (models.resnet101,       models.ResNet101_Weights.DEFAULT,
                        ["layer1", "layer2", "layer3", "layer4"]),
    "densenet121":     (models.densenet121,     models.DenseNet121_Weights.DEFAULT,
                        ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]),
    "densenet201":     (models.densenet201,     models.DenseNet201_Weights.DEFAULT,
                        ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT,
                        ["features.2", "features.3", "features.5", "features.7"]),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT,
                        ["features.2", "features.3", "features.5", "features.7"]),
    "convnext_tiny":   (models.convnext_tiny,   models.ConvNeXt_Tiny_Weights.DEFAULT,
                        ["features.1", "features.3", "features.5", "features.7"]),
    "convnext_base":   (models.convnext_base,   models.ConvNeXt_Base_Weights.DEFAULT,
                        ["features.1", "features.3", "features.5", "features.7"]),
    "vgg16":           (models.vgg16,            models.VGG16_Weights.DEFAULT,
                        ["features.8", "features.16", "features.23", "features.30"]),
    "vgg19":           (models.vgg19,            models.VGG19_Weights.DEFAULT,
                        ["features.9", "features.18", "features.27", "features.36"]),
}


def load_cnn_model(model_name, device="cuda"):
    print(f"加载 CNN 模型: {model_name}")
    if model_name not in CNN_REGISTRY:
        raise ValueError(f"不支持的 CNN 模型: {model_name}，支持: {list(CNN_REGISTRY.keys())}")

    model_fn, weights, target_layers = CNN_REGISTRY[model_name]
    base = model_fn(weights=weights).eval().to(device)

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return base, transform, target_layers, features


def extract_cnn_embeddings(model, transform, image_paths, target_layers, features, device, batch_size=4):
    """提取 CNN 模型各层 embedding (GAP)，返回 list of (N, d) arrays"""
    all_layer_feats = {k: [] for k in target_layers}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  CNN batches", leave=False):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"  跳过损坏图像: {p} ({e})")
                continue
        if not imgs:
            continue

        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            _ = model(imgs)
            for k in target_layers:
                feat = features[k]
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])  # GAP: (B,C,H,W) → (B,C)
                all_layer_feats[k].append(feat.cpu().numpy())

    return [np.concatenate(all_layer_feats[k], axis=0) for k in target_layers]


# ####################################################################
#  模型名称映射（短名称 → HuggingFace ID 或 torchvision 名称）
# ####################################################################

VIT_MODELS = {
    # ── DINOv2 ──
    "dinov2-small":  "facebook/dinov2-small",
    "dinov2-base":   "facebook/dinov2-base",
    "dinov2-large":  "facebook/dinov2-large",
    # ── DINO ──
    "dino-vitb16":   "facebook/dino-vitb16",
    "dino-vits16":   "facebook/dino-vits16",
    # ── BEiT ──
    "beit-base":     "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-large":    "microsoft/beit-large-patch16-224-pt22k-ft22k",
    # ── DeiT ──
    "deit-base":     "facebook/deit-base-patch16-224",
    "deit-small":    "facebook/deit-small-patch16-224",
    # ── ViT (Google) ──
    "vit-base":      "google/vit-base-patch16-224-in21k",
    "vit-large":     "google/vit-large-patch16-224-in21k",
    # ── ViT-MAE ──
    "vit-mae-base":  "facebook/vit-mae-base",
    "vit-mae-large": "facebook/vit-mae-large",
    # ── ViT-MSN ──
    "vit-msn-base":  "facebook/vit-msn-base",
    "vit-msn-large": "facebook/vit-msn-large",
    # ── Data2Vec ──
    "data2vec-base":  "facebook/data2vec-vision-base",
    "data2vec-large": "facebook/data2vec-vision-large",
    # ── CLIP ──
    "clip-base-32":  "openai/clip-vit-base-patch32",
    "clip-base-16":  "openai/clip-vit-base-patch16",
    "clip-large-14": "openai/clip-vit-large-patch14",
    # ── Swin ──
    "swin-tiny":     "microsoft/swin-tiny-patch4-window7-224",
    "swin-small":    "microsoft/swin-small-patch4-window7-224",
    "swin-large":    "microsoft/swin-large-patch4-window7-224",
    # ── SAM ──
    "sam-base":      "facebook/sam-vit-base",
    "sam-large":     "facebook/sam-vit-large",
    "sam-huge":      "facebook/sam-vit-huge",
}

# CNN 模型直接用 CNN_REGISTRY 的 key


def is_cnn(name):
    return name in CNN_REGISTRY


def is_vit(name):
    return name in VIT_MODELS or name.startswith(("facebook/", "google/", "openai/", "microsoft/"))


def resolve_vit_name(name):
    """短名称 → 完整 HuggingFace ID"""
    return VIT_MODELS.get(name, name)


# ####################################################################
#  主流程
# ####################################################################

def run_extraction(
    model_name,
    image_paths,
    save_root,
    device="cuda",
    batch_size=8,
    cls_only=False,
):
    """对单个模型提取所有图片的 embedding 并保存"""
    # 确定模型标签
    if is_cnn(model_name):
        model_tag = model_name
    else:
        full_name = resolve_vit_name(model_name)
        model_tag = full_name.split("/")[-1]

    save_path = os.path.join(save_root, f"{model_tag}.npy")
    if os.path.exists(save_path):
        print(f"  已存在，跳过: {save_path}")
        return

    # ── 提取 ──
    if is_cnn(model_name):
        model, transform, target_layers, features = load_cnn_model(model_name, device)
        X_layers = extract_cnn_embeddings(
            model, transform, image_paths, target_layers, features,
            device=device, batch_size=batch_size,
        )
        del model, features
    else:
        full_name = resolve_vit_name(model_name)
        extractor, model = load_vit_model(full_name, device)
        X_layers = extract_vit_embeddings(
            extractor, model, image_paths,
            device=device, cls_only=cls_only, batch_size=batch_size,
        )
        del extractor, model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not X_layers:
        print(f"  ⚠ 没有成功提取到 embedding")
        return

    # ── 对齐维度 + 保存 ──
    X_layers, target_dim = align_layer_dims(X_layers)
    # (n_layers, n_images, feat_dim)
    X_all = np.stack(X_layers, axis=0).astype(np.float16)
    np.save(save_path, X_all)
    print(f"  ✅ shape={X_all.shape}  →  {save_path}")


# ====================================================================
#  CLI
# ====================================================================

ALL_MODEL_NAMES = list(VIT_MODELS.keys()) + list(CNN_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一图像 Embedding 提取（ViT + CNN）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="图片数据集根目录（支持子文件夹或平铺）")
    parser.add_argument("--save_root", type=str, default="embeddings",
                        help="输出目录，每个模型一个 .npy 文件")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备: cuda / mps / cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cls_only", action="store_true",
                        help="ViT 模型仅用 CLS token（默认 patch mean）")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"要跑的模型名称列表，不指定则跑全部。\n可选: {ALL_MODEL_NAMES}")
    return parser.parse_args()


def main():
    args = parse_args()

    # 收集图片
    image_paths = collect_image_paths(args.data_root)
    if len(image_paths) == 0:
        print("未找到任何图片，退出。")
        return

    os.makedirs(args.save_root, exist_ok=True)

    # 确定要跑的模型
    if args.models:
        model_list = args.models
    else:
        model_list = ALL_MODEL_NAMES

    # 保存图片路径索引，方便后续查找
    index_path = os.path.join(args.save_root, "image_paths.txt")
    with open(index_path, "w") as f:
        for p in image_paths:
            f.write(p + "\n")
    print(f"图片索引已保存: {index_path}")

    # 逐模型提取
    for model_name in model_list:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            run_extraction(
                model_name=model_name,
                image_paths=image_paths,
                save_root=args.save_root,
                device=args.device,
                batch_size=args.batch_size,
                cls_only=args.cls_only,
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n全部完成！输出目录: {args.save_root}")


if __name__ == "__main__":
    main()