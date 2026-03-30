import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import gc
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob

import torch
from torchvision import models, transforms
from PIL import Image


# =========================
# 1️⃣ CNN 模型加载 + hook
# =========================
def load_cnn_model_multilayer(model_name="resnet50", device="mps"):

    if model_name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        target_layers = ["layer1", "layer2", "layer3", "layer4"]

    elif model_name == "resnet34":
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        target_layers = ["layer1", "layer2", "layer3", "layer4"]

    elif model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        target_layers = ["layer1", "layer2", "layer3", "layer4"]

    elif model_name == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        target_layers = ["layer1", "layer2", "layer3", "layer4"]

    elif model_name == "wide_resnet50_2":
        base = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        target_layers = ["layer1", "layer2", "layer3", "layer4"]

    elif model_name == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        target_layers = ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]

    elif model_name == "densenet201":
        base = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        target_layers = ["features.denseblock1", "features.denseblock2",
                         "features.denseblock3", "features.denseblock4"]

    elif model_name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        target_layers = ["features.2", "features.3", "features.5", "features.7"]

    elif model_name == "efficientnet_b4":
        base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        target_layers = ["features.2", "features.3", "features.5", "features.7"]

    elif model_name == "convnext_tiny":
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        target_layers = ["features.1", "features.3", "features.5", "features.7"]

    elif model_name == "convnext_base":
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        target_layers = ["features.1", "features.3", "features.5", "features.7"]

    elif model_name == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        target_layers = ["features.8", "features.16", "features.23", "features.30"]

    elif model_name == "vgg19":
        base = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        target_layers = ["features.9", "features.18", "features.27", "features.36"]

    else:
        raise ValueError(f"{model_name} not supported")

    base.eval().to(device)

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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return base, transform, target_layers, features


# =========================
# 2️⃣ 手动随机投影（降维到最小层维度）
# =========================
def random_project(X, target_dim, random_state=42):
    d = X.shape[1]
    if d == target_dim:
        return X.astype(np.float32)
    rng = np.random.RandomState(random_state)
    R = rng.randn(d, target_dim).astype(np.float32) / np.sqrt(target_dim)
    return X.astype(np.float32) @ R


def get_cnn_multilayer_embeddings(
    model, transform, img_paths,
    target_layers, features,
    device="mps", batch_size=4,
):
    all_layer_feats = {k: [] for k in target_layers}

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        imgs = torch.stack(imgs).to(device)

        with torch.no_grad():
            _ = model(imgs)
            for k in target_layers:
                feat = features[k].mean(dim=[2, 3])  # GAP → (B, D)
                all_layer_feats[k].append(feat.cpu().numpy())

    raw_layers = []
    for k in target_layers:
        X = np.concatenate(all_layer_feats[k], axis=0)
        raw_layers.append(X)

    # target_dim = 最小层维度，所有层降维到同一紧凑空间
    target_dim = min(X.shape[1] for X in raw_layers)
    print(f"   layer dims: {[X.shape[1] for X in raw_layers]} → target_dim={target_dim}")

    X_layers = [random_project(X, target_dim) for X in raw_layers]

    return X_layers  # list of (n_samples, target_dim)，所有层维度一致


# =========================
# 3️⃣ 主 pipeline
# =========================
def generate_cnn_embeddings(
    model_name="resnet50",
    data_root="data/image_data/ds004192-download",
    img_root="data/image_data/images",
    save_root="filterData/img/design_matrix_cnn",
    tr=2.0,
    device="mps",
    batch_size=4,
):
    model_save_root = os.path.join(save_root, model_name)
    os.makedirs(model_save_root, exist_ok=True)

    model, transform, target_layers, features = load_cnn_model_multilayer(model_name, device)
    print(f"Loaded: {model_name} | layers: {target_layers}")

    subs = sorted(glob(os.path.join(data_root, "sub-*")))

    def process_run(events_file, sub, ses):
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

        X_layers = get_cnn_multilayer_embeddings(
            model, transform, img_paths,
            target_layers, features,
            device=device,
            batch_size=batch_size,
        )

        n_layers = len(X_layers)
        feat_dim = X_layers[0].shape[1]  # 所有层维度一致

        bold_file = events_file.replace("_events.tsv", "_bold.nii.gz")
        if not os.path.exists(bold_file):
            return

        n_tr = nib.load(bold_file).shape[-1]
        df["tr_idx"] = (df["onset"] / tr).round().astype(int)

        X_all = np.zeros((n_layers, n_tr, feat_dim), dtype=np.float32)

        for li in range(n_layers):
            for si, row in df.iterrows():
                ti = row["tr_idx"]
                if 0 <= ti < n_tr:
                    X_all[li, ti] = X_layers[li][si]

        X_all = X_all.astype(np.float16)

        sub_save_dir = os.path.join(model_save_root, sub, ses)
        os.makedirs(sub_save_dir, exist_ok=True)

        bold_name = os.path.basename(events_file).replace("_events.tsv", "")
        save_path = os.path.join(sub_save_dir, bold_name + "_bold_embedding.npy")

        np.save(save_path, X_all)
        print(f"  Saved: {save_path}  shape={X_all.shape}")

    for sub_path in subs:
        sub = os.path.basename(sub_path)
        ses_list = sorted(glob(os.path.join(sub_path, "ses-things*")))

        for ses_path in ses_list:
            ses = os.path.basename(ses_path)
            func_dir = os.path.join(ses_path, "func")

            event_files = sorted(glob(os.path.join(func_dir, "*_events.tsv")))
            for ef in event_files:
                process_run(ef, sub, ses)

    print(f"\nDone: {model_save_root}")

    # 释放显存
    del model
    del features
    gc.collect()
    torch.mps.empty_cache()


# =========================
# 4️⃣ 批量运行
# =========================
if __name__ == "__main__":
    cnn_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "wide_resnet50_2",
        "densenet121",
        "densenet201",
        "efficientnet_b0",
        "efficientnet_b4",
        "convnext_tiny",
        "convnext_base",
        "vgg16",
        "vgg19",
    ]

    for model_name in cnn_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_cnn_embeddings(
                model_name=model_name,
                device="mps",
                batch_size=4
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            torch.mps.empty_cache()