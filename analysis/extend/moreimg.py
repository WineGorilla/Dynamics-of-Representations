import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import gc
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from models.image_encoder import load_image_model, get_image_embeddings


def generate_image_embeddings(
    model_name,
    data_root="data/image_data/ds004192-download",
    img_root="data/image_data/images",
    save_root="filterData/img/design_matrix_extra",
    tr=2.0,
    device="mps",
    batch_size=8,
):
    model_tag = model_name.split("/")[-1]
    model_save_root = os.path.join(save_root, model_tag)
    os.makedirs(model_save_root, exist_ok=True)

    extractor, model = load_image_model(model_name, device=device)

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

        X_layers = get_image_embeddings(
            extractor, model, img_paths,
            device=device, all_layers=True, cls_only=True,
            batch_size=batch_size
        )
        n_layers = len(X_layers)
        feat_dim = X_layers[0].shape[1]

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

    # 释放模型显存
    del model
    del extractor
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    extra_models = [
        # # ── CLIP 系列 ──────────────────────────────────
        # "openai/clip-vit-base-patch32",
        # "openai/clip-vit-base-patch16",
        # "openai/clip-vit-large-patch14",
        # "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        # "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",

        # # ── Swin Transformer 系列 ──────────────────────
        # "microsoft/swin-tiny-patch4-window7-224",
        # "microsoft/swin-small-patch4-window7-224",
        # "microsoft/swin-base-patch4-window7-224",
        # "microsoft/swin-large-patch4-window7-224",

        # # ── SigLIP 系列 ────────────────────────────────
        # "google/siglip-base-patch16-224",
        # "google/siglip-large-patch16-384",

        # # ── SAM 视觉编码器 ─────────────────────────────
        "facebook/sam-vit-base",
        # "facebook/sam-vit-large",
        # "facebook/sam-vit-huge",
    ]

    for model_name in extra_models:
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        try:
            generate_image_embeddings(
                model_name=model_name,
                device="mps",
                batch_size=8
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        finally:
            gc.collect()
            torch.mps.empty_cache()