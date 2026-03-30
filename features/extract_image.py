import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from core.encoder.image_encoder import load_image_model, get_image_embeddings


def generate_image_embeddings(
    model_name="google/vit-base-patch16-224",
    data_root="data/image_data/ds004192-download",
    img_root="data/image_data/images",
    save_root="filterData/img/design_matrix",
    tr=2.0,
    device="mps",
    batch_size=8,
):

    # 保存目录
    model_tag = model_name.split("/")[-1]
    model_save_root = os.path.join(save_root, model_tag)
    os.makedirs(model_save_root, exist_ok=True)

    extractor, model = load_image_model(model_name, device=device)

    subs = sorted(glob(os.path.join(data_root, "sub-*")))

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
        save_name = bold_name + "_bold_embedding.npy"
        save_path = os.path.join(sub_save_dir, save_name)

        np.save(save_path, X_all)
        print(f"Saved (FP16): {save_path}")

    for sub_path in subs:
        sub = os.path.basename(sub_path)
        ses_list = sorted(glob(os.path.join(sub_path, "ses-things*")))

        for ses_path in ses_list:
            ses = os.path.basename(ses_path)
            func_dir = os.path.join(ses_path, "func")

            event_files = sorted(glob(os.path.join(func_dir, "*_events.tsv")))
            for ef in event_files:
                run_tag = [x for x in ef.split("_") if "run" in x][0]
                process_run(ef, sub, ses, run_tag)

    print(f"\nAll Done! Saved in: {model_save_root}")


# 调用
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
    for model_name in vision_models:
        generate_image_embeddings(
            model_name=model_name,
            device="mps",
            batch_size=8
        )
