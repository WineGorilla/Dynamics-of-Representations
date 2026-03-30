import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
import pandas as pd


def extract_roi_timeseries(
        fmri_path,
        atlas_path,
        label_names=None,
        modality_name="unknown",
        tr=2.0,
        save=True):

    import numpy as np
    from nilearn.maskers import NiftiLabelsMasker


    fmri_img = fmri_path
    atlas_img = atlas_path


    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        t_r=tr
    )


    roi_ts = masker.fit_transform(fmri_img)  # (T, 200)

    if save:
        np.save(f"{modality_name}.npy", roi_ts)
        print(f"已保存: {modality_name}.npy")

    return roi_ts




def extract_roi_signals(fmri_path, atlas_img, label_names, modality_name):

    fmri_raw = nib.load(fmri_path)

    fmri_img = resample_to_img(
        fmri_raw,
        atlas_img,
        interpolation="continuous"
    )

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        t_r=2.0
    )

    # ROI 提取
    roi_ts = masker.fit_transform(fmri_img)

    # 如果无有效 ROI
    if roi_ts.size == 0:
        print("Warning: no ROI extracted for", modality_name)
        return None

    # atlas.labels[0] 是背景
    valid_labels = label_names[1: roi_ts.shape[1] + 1]

    df = pd.DataFrame(roi_ts, columns=valid_labels)
    df.to_csv(f"filterData/shared_masks/{modality_name}_roi_timeseries.csv", index=False)
    return df
