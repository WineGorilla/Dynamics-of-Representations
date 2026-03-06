import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import pandas as pd

from nilearn import datasets


atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=200,
    yeo_networks=7,
    resolution_mm=2
)

labels = atlas.labels


roi_labels = labels[1:]

print("Number of ROI labels:", len(roi_labels))
print("Example labels:", roi_labels[:5])

# 路径设置

root_dir = "results/dmd_mean/hrf"
modalities = {
    "lang": os.path.join(root_dir, "lang_fused"),
    "audio": os.path.join(root_dir, "audio_fused"),
    "vision": os.path.join(root_dir, "img_fused"),
}

records = []

for modality, path_root in modalities.items():

    for model_name in sorted(os.listdir(path_root)):
        model_dir = os.path.join(path_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        roi_path = os.path.join(model_dir, "group_mean_roi.npy")
        if not os.path.exists(roi_path):
            continue

        arr = np.load(roi_path).astype(float)
        arr = np.squeeze(arr)

        if arr.shape[0] != len(roi_labels):
            print(f"[Skip] ROI mismatch: {model_name}")
            continue

        size = (
            "base" if "base" in model_name
            else "large" if "large" in model_name
            else "other"
        )
        family = model_name.split("-")[0].lower()

        rec = {
            "model_name": model_name,
            "modality": modality,
            "family": family,
            "size": size,
        }
        rec.update({roi: val for roi, val in zip(roi_labels, arr)})
        records.append(rec)



# 保存原始数据，每一个模型的每一个roi的pearson
df_raw = pd.DataFrame(records)

os.makedirs("processed/dmd_mean/hrf", exist_ok=True)


df_raw.to_csv(
    "processed/dmd_mean/hrf/BrainModelCovDataset_raw.csv",
    index=False
)

print("Saved RAW dataset")
print(df_raw.head(3))

