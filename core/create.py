# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import os
# import numpy as np
# import pandas as pd

# from nilearn import datasets


# atlas = datasets.fetch_atlas_schaefer_2018(
#     n_rois=200,
#     yeo_networks=7,
#     resolution_mm=2
# )

# labels = atlas.labels


# roi_labels = labels[1:]

# print("Number of ROI labels:", len(roi_labels))
# print("Example labels:", roi_labels[:5])

# # 路径设置

# root_dir = "results/soft/01"
# modalities = {
#     "lang": os.path.join(root_dir, "lang_fused"),
#     "audio": os.path.join(root_dir, "audio_fused"),
#     "vision": os.path.join(root_dir, "img_fused"),
# }

# records = []

# for modality, path_root in modalities.items():

#     for model_name in sorted(os.listdir(path_root)):
#         model_dir = os.path.join(path_root, model_name)
#         if not os.path.isdir(model_dir):
#             continue

#         roi_path = os.path.join(model_dir, "group_mean_roi.npy")
#         if not os.path.exists(roi_path):
#             continue

#         arr = np.load(roi_path).astype(float)
#         arr = np.squeeze(arr)

#         if arr.shape[0] != len(roi_labels):
#             print(f"[Skip] ROI mismatch: {model_name}")
#             continue

#         size = (
#             "base" if "base" in model_name
#             else "large" if "large" in model_name
#             else "other"
#         )
#         family = model_name.split("-")[0].lower()

#         rec = {
#             "model_name": model_name,
#             "modality": modality,
#             "family": family,
#             "size": size,
#         }
#         rec.update({roi: val for roi, val in zip(roi_labels, arr)})
#         records.append(rec)



# # 保存原始数据，每一个模型的每一个roi的pearson
# df_raw = pd.DataFrame(records)

# os.makedirs("processed/soft/hrf/01", exist_ok=True)


# df_raw.to_csv(
#     "processed/soft/hrf/01/BrainModelCovDataset_raw.csv",
#     index=False
# )

# print("Saved RAW dataset")
# print(df_raw.head(3))



import os
import numpy as np
import pandas as pd
from nilearn import datasets


def build_brain_model_dataset(center="01",
                              results_root="results/pearson_dfc",
                              save_root="processed/pearson_dfc/hrf"):
    
    # =========================
    # Load Schaefer Atlas
    # =========================
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200,
        yeo_networks=7,
        resolution_mm=2
    )

    labels = atlas.labels
    roi_labels = labels[1:]

    print("Number of ROI labels:", len(roi_labels))
    print("Example labels:", roi_labels[:5])

    # =========================
    # Paths
    # =========================
    root_dir = os.path.join(results_root, center)

    modalities = {
        "lang": os.path.join(root_dir, "lang_fused"),
        "audio": os.path.join(root_dir, "audio_fused"),
        "vision": os.path.join(root_dir, "img_fused"),
    }

    records = []

    # =========================
    # Collect data
    # =========================
    for modality, path_root in modalities.items():

        if not os.path.exists(path_root):
            print(f"[Skip] modality path not found: {path_root}")
            continue

        for model_name in sorted(os.listdir(path_root)):

            model_dir = os.path.join(path_root, model_name)
            if not os.path.isdir(model_dir):
                continue

            roi_path = os.path.join(model_dir, "roi_score.npy")
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


    # Save dataset
    df_raw = pd.DataFrame(records)

    save_dir = os.path.join(save_root, center)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "BrainModelDataset_raw.csv")
    df_raw.to_csv(save_path, index=False)

    print("Saved RAW dataset:", save_path)
    print(df_raw.head(3))

    return df_raw


#centers = ["01","02","03","04","05","06","07","08","09","10","12","14"]
centers = ['10']
for c in centers:
    build_brain_model_dataset(center=c)