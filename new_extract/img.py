import torch
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoImageProcessor,
    CLIPModel, SamModel
)
from PIL import Image
import numpy as np
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=2 python new_extract/img.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
import pandas as pd
import nibabel as nib
from glob import glob

# 没有 CLS token 的模型类型
NO_CLS_TYPES = {"swin", "sam", "siglip"}


def load_image_model(model_name="facebook/dinov2-base", device="mps"):
    print(f"加载图像模型: {model_name}")
    try:
        extractor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)

    model_lower = model_name.lower()

    if "clip" in model_lower or "siglip" in model_lower:
        # CLIP/SigLIP: 只取视觉编码器
        full_model = CLIPModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.vision_model
        model.config = full_model.config.vision_config
        model.config.model_type = "clip_vision"

    elif "sam" in model_lower:
        # SAM: 只取视觉编码器
        full_model = SamModel.from_pretrained(model_name, output_hidden_states=True)
        model = full_model.vision_encoder
        model.config = full_model.config.vision_config
        model.config.model_type = "sam"

    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model = model.to(device)
    model.eval()
    return extractor, model


def _pool_hidden(hidden_states, model, cls_only):
    """根据模型类型选择池化方式"""
    model_type = getattr(model.config, "model_type", "")
    has_cls = model_type not in NO_CLS_TYPES

    if cls_only:
        if not has_cls:
            print(f"{model_type} 没有 CLS token，自动切换为 patch mean")
            return [h.mean(dim=1).detach() for h in hidden_states]
        return [h[:, 0, :].detach() for h in hidden_states]
    else:
        if has_cls:
            return [h[:, 1:, :].mean(dim=1).detach() for h in hidden_states]
        else:
            # SAM 的 hidden state 可能是 4D: (B, H, W, D)
            result = []
            for h in hidden_states:
                if h.dim() == 4:
                    # (B, H, W, D) -> (B, D)
                    result.append(h.mean(dim=[1, 2]).detach())
                else:
                    # (B, N, D) -> (B, D)
                    result.append(h.mean(dim=1).detach())
            return result


def random_project(X, target_dim, random_state=42):
    """随机投影降维，近似保持距离结构"""
    d = X.shape[1]
    if d == target_dim:
        return X.astype(np.float32)
    rng = np.random.RandomState(random_state)
    R = rng.randn(d, target_dim).astype(np.float32) / np.sqrt(target_dim)
    return X.astype(np.float32) @ R


def _forward_model(model, inputs):
    """根据模型类型选择前向传播方式"""
    model_type = getattr(model.config, "model_type", "")

    if model_type == "sam":
        # SAM vision encoder 直接接受 pixel_values
        outputs = model(inputs["pixel_values"], output_hidden_states=True)
    elif model_type == "clip_vision":
        # CLIP vision model 只需要 pixel_values
        outputs = model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
    else:
        outputs = model(**inputs)

    return outputs


def get_image_embeddings(extractor, model, image_paths, device, all_layers=True, cls_only=False, batch_size=4):
    n_layers = None
    layer_collectors = {}
    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
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


def get_image_embeddings_from_pil(extractor, model, images, device, cls_only=False, batch_size=4):
    """
    直接接受 PIL Image 列表，无需临时文件。
    返回 list of (N, d) arrays，长度 = 层数。
    """
    n_layers = None
    layer_collectors = {}
    model.eval()

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        inputs = extractor(images=batch_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = _forward_model(model, inputs)
            hidden_states = outputs.hidden_states

        layer_embeds = _pool_hidden(hidden_states, model, cls_only)

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


def generate_image_embeddings(
    model_name="google/vit-base-patch16-224",
    data_root="data/img_data/ds004192-download",
    img_root="data/img_data/images",
    save_root="filterData/img/design_matrix",
    tr=2.0,
    device="cuda",
    batch_size=8,
):
    model_tag = model_name.split("/")[-1]
    model_save_root = os.path.join(save_root, model_tag)
    os.makedirs(model_save_root, exist_ok=True)

    extractor, model = load_image_model(model_name, device=device)

    subs = sorted(glob(os.path.join(data_root, "sub-*")))

    # 先统计所有 run
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

        X_layers = get_image_embeddings(
            extractor, model, img_paths,
            device=device, all_layers=True, cls_only=False,
            batch_size=batch_size
        )

        if len(X_layers) == 0:
            return

        n_layers = len(X_layers)

        bold_file = events_file.replace("_events.tsv", "_bold.nii.gz")
        if not os.path.exists(bold_file):
            return
        n_tr = nib.load(bold_file).shape[-1]

        df["tr_idx"] = (df["onset"] / tr).round().astype(int)

        # 统一维度：如果各层维度不同，用随机投影降维到最小维度
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

        X_all = X_all.astype(np.float16)

        sub_save_dir = os.path.join(model_save_root, sub, ses)
        os.makedirs(sub_save_dir, exist_ok=True)

        bold_name = os.path.basename(events_file).replace("_events.tsv", "")
        save_name = bold_name + "_bold_embedding.npy"
        save_path = os.path.join(sub_save_dir, save_name)

        np.save(save_path, X_all)

    # 一个干净的总进度条
    for ef, sub, ses, run_tag in tqdm(all_runs, desc=f"{model_tag}", ncols=80):
        process_run(ef, sub, ses, run_tag)

    del extractor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll Done! Saved in: {model_save_root}")


if __name__ == "__main__":
    vision_models = [
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
    ]

    extra_models = [
        # ── CLIP 系列 ──────────────────────────────────
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",

        # ── Swin Transformer 系列 ──────────────────────
        "microsoft/swin-tiny-patch4-window7-224",
        "microsoft/swin-small-patch4-window7-224",
        "microsoft/swin-large-patch4-window7-224",


        # ── SAM 视觉编码器 ─────────────────────────────
        "facebook/sam-vit-base",
        "facebook/sam-vit-large",
        "facebook/sam-vit-huge",
    ]

    all_models = extra_models + vision_models

    for model_name in all_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_image_embeddings(
                model_name=model_name,
                device="cuda",
                batch_size=4
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()