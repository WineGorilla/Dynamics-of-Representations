"""
fMRI 视觉 Embedding 提取（ViT + CNN 合并，36个模型）
=====================================================
在保存前过滤掉全零 TR，只保留有图片刺激的时间点。

用法:
  CUDA_VISIBLE_DEVICES=0 python extractnew/extract_vision_fmri.py --device cuda --batch_size 4
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
import gc
import ssl
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import models, transforms
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoImageProcessor,
    CLIPModel, SamModel,
)

ssl._create_default_https_context = ssl._create_unverified_context


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


# ####################################################################
#  ViT 模型
# ####################################################################

NO_CLS_TYPES = {"swin", "sam", "siglip"}


def load_image_model(model_name, device="cuda"):
    print(f"加载图像模型: {model_name}")
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


def _forward_model(model, inputs):
    model_type = getattr(model.config, "model_type", "")
    if model_type == "sam":
        return model(inputs["pixel_values"], output_hidden_states=True)
    elif model_type == "clip_vision":
        return model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
    else:
        return model(**inputs)


def _pool_hidden(hidden_states, model, cls_only):
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


def get_image_embeddings(extractor, model, image_paths, device, all_layers=True, cls_only=False, batch_size=4):
    n_layers = None
    layer_collectors = {}
    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"跳过损坏图像: {p} ({e})")
                continue

        if len(batch_imgs) == 0:
            continue

        try:
            inputs = extractor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = _forward_model(model, inputs)
                hidden_states = outputs.hidden_states
            layer_embeds = _pool_hidden(hidden_states, model, cls_only)
        except Exception as e:
            print(f"Batch 失败，逐张重试: {e}")
            layer_embeds_list = []
            for img in batch_imgs:
                try:
                    inputs = extractor(images=[img], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = _forward_model(model, inputs)
                        hidden_states = outputs.hidden_states
                    layer_embeds_list.append(_pool_hidden(hidden_states, model, cls_only))
                except Exception as e2:
                    print(f"跳过问题图像: {e2}")
                    continue
            if len(layer_embeds_list) == 0:
                continue
            n_l = len(layer_embeds_list[0])
            layer_embeds = [torch.cat([le[li] for le in layer_embeds_list], dim=0) for li in range(n_l)]

        if n_layers is None:
            n_layers = len(layer_embeds)
            for li in range(n_layers):
                layer_collectors[li] = []

        for li in range(n_layers):
            layer_collectors[li].append(layer_embeds[li].cpu().numpy())

    if n_layers is None:
        return []

    X_layers = [np.concatenate(layer_collectors[li], axis=0) for li in range(n_layers)]
    return X_layers


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


def get_cnn_multilayer_embeddings(model, transform, img_paths, target_layers, features, device, batch_size=4):
    all_layer_feats = {k: [] for k in target_layers}

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"跳过损坏图像: {p} ({e})")
                continue
        if not imgs:
            continue

        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            _ = model(imgs)
            for k in target_layers:
                feat = features[k]
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                all_layer_feats[k].append(feat.cpu().numpy())

    return [np.concatenate(all_layer_feats[k], axis=0) for k in target_layers]


# ####################################################################
#  模型列表 (36个: 26 ViT + 10 CNN)
# ####################################################################

VIT_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "facebook/data2vec-vision-base",
    "facebook/data2vec-vision-large",
    "facebook/deit-base-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/dino-vitb16",
    "facebook/dino-vits16",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-small",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-msn-base",
    "facebook/vit-msn-large",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "microsoft/swin-tiny-patch4-window7-224",
    "microsoft/swin-small-patch4-window7-224",
    "microsoft/swin-large-patch4-window7-224",
    "facebook/sam-vit-base",
    "facebook/sam-vit-large",
    "facebook/sam-vit-huge",
]

CNN_MODELS = list(CNN_REGISTRY.keys())

ALL_MODELS = VIT_MODELS + CNN_MODELS


# ####################################################################
#  Embedding 提取主流程
# ####################################################################

def generate_embeddings(
    model_name,
    data_root="data/img_data/ds004192-download",
    img_root="data/img_data/images",
    save_root="filterData/img/design_matrix",
    tr=2.0,
    device="cuda",
    batch_size=4,
):
    model_tag = model_name.split("/")[-1]
    model_save_root = os.path.join(save_root, model_tag)
    os.makedirs(model_save_root, exist_ok=True)

    is_cnn = model_name in CNN_REGISTRY

    if is_cnn:
        model, transform, target_layers, features = load_cnn_model(model_name, device)
    else:
        extractor, model = load_image_model(model_name, device)

    subs = sorted(glob(os.path.join(data_root, "sub-*")))

    all_runs = []
    for sub_path in subs:
        sub = os.path.basename(sub_path)
        ses_list = sorted(glob(os.path.join(sub_path, "ses-things*")))
        for ses_path in ses_list:
            ses = os.path.basename(ses_path)
            func_dir = os.path.join(ses_path, "func")
            event_files = sorted(glob(os.path.join(func_dir, "*_events.tsv")))
            for ef in event_files:
                run_tag = [x for x in ef.split("_") if "run" in x][0]
                all_runs.append((ef, sub, ses, run_tag))

    def process_run(events_file, sub, ses, run_tag):
        df = pd.read_csv(events_file, sep="\t")
        df = df[df["trial_type"].isin(["exp", "test"])].reset_index(drop=True)
        if len(df) == 0:
            return

        valid_rows, img_paths = [], []
        for _, row in df.iterrows():
            if isinstance(row.get("file_path", None), str):
                img_path = os.path.join(img_root, row["file_path"])
                if os.path.exists(img_path):
                    valid_rows.append(row)
                    img_paths.append(img_path)

        df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(df) == 0:
            return

        if is_cnn:
            X_layers = get_cnn_multilayer_embeddings(
                model, transform, img_paths, target_layers, features,
                device=device, batch_size=batch_size,
            )
        else:
            X_layers = get_image_embeddings(
                extractor, model, img_paths,
                device=device, all_layers=True, cls_only=False,
                batch_size=batch_size,
            )

        if len(X_layers) == 0:
            return

        n_layers = len(X_layers)

        bold_file = events_file.replace("_events.tsv", "_bold.nii.gz")
        if not os.path.exists(bold_file):
            return
        n_tr = nib.load(bold_file).shape[-1]

        df["tr_idx"] = (df["onset"] / tr).round().astype(int)

        # 统一维度
        feat_dims = [X_layers[li].shape[1] for li in range(n_layers)]
        target_dim = min(feat_dims)
        if len(set(feat_dims)) > 1:
            print(f"  层维度不一致 {feat_dims} → 随机投影到 {target_dim}")
            X_layers = [random_project(X, target_dim) for X in X_layers]

        X_all = np.zeros((n_layers, n_tr, target_dim), dtype=np.float32)
        for li in range(n_layers):
            for si, row in df.iterrows():
                ti = row["tr_idx"]
                if 0 <= ti < n_tr:
                    X_all[li, ti] = X_layers[li][si]

        # ── 过滤全零 TR ──
        # 对所有层求和，只保留至少有一层非零的时间点
        nonzero_mask = np.abs(X_all).sum(axis=(0, 2)) > 0  # (n_tr,)
        X_filtered = X_all[:, nonzero_mask, :]

        X_filtered = X_filtered.astype(np.float16)

        sub_save_dir = os.path.join(model_save_root, sub, ses)
        os.makedirs(sub_save_dir, exist_ok=True)

        bold_name = os.path.basename(events_file).replace("_events.tsv", "")
        save_name = bold_name + "_bold_embedding.npy"
        save_path = os.path.join(sub_save_dir, save_name)
        np.save(save_path, X_filtered)

    for ef, sub, ses, run_tag in tqdm(all_runs, desc=f"{model_tag}", ncols=80):
        process_run(ef, sub, ses, run_tag)

    if is_cnn:
        del model, features
    else:
        del extractor, model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  ✅ Done: {model_save_root}")


# ####################################################################
#  CLI
# ####################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="fMRI 视觉 Embedding 提取 (36个模型，过滤全零TR)",
    )
    parser.add_argument("--data_root", type=str,
                        default="data/img_data/ds004192-download")
    parser.add_argument("--img_root", type=str,
                        default="data/img_data/images")
    parser.add_argument("--save_root", type=str,
                        default="embedding/fmriimg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--models", nargs="*", default=None,
                        help="指定模型列表，不指定则跑全部36个")
    return parser.parse_args()


def main():
    args = parse_args()
    model_list = args.models if args.models else ALL_MODELS

    print(f"共 {len(model_list)} 个模型")

    for model_name in model_list:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            generate_embeddings(
                model_name=model_name,
                data_root=args.data_root,
                img_root=args.img_root,
                save_root=args.save_root,
                tr=args.tr,
                device=args.device,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n全部完成！")


if __name__ == "__main__":
    main()