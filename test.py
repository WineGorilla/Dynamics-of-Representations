import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image, surface, plotting

# 读取多模态模型roi数据集
path = "processed/dmd_mean/hrf/BrainModelCovDataset_raw.csv"
df = pd.read_csv(path)

meta_cols = ["model_name", "modality", "family", "size"]
roi_cols = [c for c in df.columns if c not in meta_cols]

print("Number of ROIs:", len(roi_cols))
print("Modalities:", df["modality"].unique())

out_dir = "processed/dmd_mean/hrf/brain_score_by_modality_sigmoid"
os.makedirs(out_dir, exist_ok=True)

# min152空间，roi加载
mni = datasets.load_mni152_template()

atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=200, yeo_networks=7, resolution_mm=2
)
atlas_img = nib.load(atlas.maps)

atlas_mni = image.resample_to_img(
    atlas_img,
    mni,
    interpolation="nearest",
    force_resample=True,
    copy_header=True
)
atlas_mni_data = atlas_mni.get_fdata().astype(int)

fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")


eps = 1e-6

for modality, df_mod in df.groupby("modality"):

    print(f"\nProcessing modality: {modality}")

    r = df_mod[roi_cols].values.astype(float)

    r2 = r ** 2

    # 跨模型统计 同时考虑每一个roi的均值与方差
    mu = r2.mean(axis=0)                 # signal
    sigma = r2.std(axis=0, ddof=1) + eps # noise

    stability = mu / sigma               # SNR-like quantity

    # z-score across ROIs
    z = (stability - stability.mean()) / (stability.std() + eps)

    # sigmoid
    brain_score = 1 / (1 + np.exp(-z))

    # 保存 CSV
    df_score = pd.DataFrame({
        "feature": roi_cols,
        "brain_score": brain_score
    })

    df_score = df_score.sort_values(
        "brain_score", ascending=False
    ).reset_index(drop=True)

    df_score["rank"] = np.arange(1, len(df_score) + 1)
    df_score = df_score[["rank", "feature", "brain_score"]]

    out_csv = os.path.join(out_dir, f"{modality}_brain_score_sigmoid.csv")
    df_score.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(df_score.head(10))


    # 画 brain map
    roi_values = brain_score

    roi_map = np.zeros_like(atlas_mni_data, dtype=float)
    for idx in range(1, 201):
        roi_map[atlas_mni_data == idx] = roi_values[idx - 1]

    roi_img = nib.Nifti1Image(roi_map, affine=mni.affine)
    roi_img_mni = image.resample_to_img(
        roi_img,
        mni,
        interpolation="nearest",
        force_resample=True,
        copy_header=True
    )

    tex_left  = surface.vol_to_surf(roi_img_mni, fsaverage.pial_left)
    tex_right = surface.vol_to_surf(roi_img_mni, fsaverage.pial_right)

    plotting.plot_surf_stat_map(
        fsaverage.infl_left,
        tex_left,
        hemi="left",
        bg_map=fsaverage.sulc_left,
        cmap="viridis",
        vmin=0, vmax=1,
        colorbar=True,
        title=f"{modality.capitalize()} Brain Score (Left)"
    )

    plotting.plot_surf_stat_map(
        fsaverage.infl_right,
        tex_right,
        hemi="right",
        bg_map=fsaverage.sulc_right,
        cmap="viridis",
        vmin=0, vmax=1,
        colorbar=True,
        title=f"{modality.capitalize()} Brain Score (Right)"
    )

    plotting.show()
