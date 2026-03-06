import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ProcessPoolExecutor
from glob import glob
import nibabel as nib
from nilearn import image, datasets
from nilearn.datasets import fetch_atlas_schaefer_2018
from utils.roi_process import extract_roi_timeseries
import tempfile


# 配置路径
root_dir = "data/image_data/ds004192-download"
save_root = "filterData/img/fmri"
os.makedirs(save_root, exist_ok=True)

mni_img_worker = None
atlas_img_worker = None


# Worker 初始化
def init_worker(mni_path, atlas_path):
    global mni_img_worker, atlas_img_worker
    print("[Worker init] Loading MNI & Atlas...")

    mni_img_worker = nib.load(mni_path)
    atlas_img_worker = nib.load(atlas_path)


def load_and_resample_to_mni(fmri_path):
    global mni_img_worker
    fmri_img = nib.load(fmri_path)

    return image.resample_to_img(
        fmri_img,
        mni_img_worker,
        interpolation="nearest",
        force_resample=True,
        copy_header=True
    )

# worker主任务
def process_bold(bold_path):
    global atlas_img_worker, save_root

    try:
        parts = bold_path.split("/")
        sub_name = parts[-4]
        ses_name = parts[-3]

        out_dir = os.path.join(save_root, sub_name, ses_name)
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.basename(bold_path).replace(".nii.gz", "")
        out_path = os.path.join(out_dir, f"{fname}_shared.npy")

        if os.path.exists(out_path):
            print(f"[Skip] {bold_path}")
            return

        print(f"[Run] {bold_path}")

        fmri_mni = load_and_resample_to_mni(bold_path)

        extract_roi_timeseries(
            fmri_path=fmri_mni,
            atlas_path=atlas_img_worker,
            modality_name=out_path.replace(".npy", ""),
            tr=2.0,
            save=True
        )

    except Exception as e:
        print(f"[Error] {bold_path}: {e}")


if __name__ == "__main__":

    print("Loading MNI152 template in main process...")
    mni_img = datasets.load_mni152_template()

    tmp_mni = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(mni_img, tmp_mni.name)
    mni_path = tmp_mni.name
    print("MNI saved to:", mni_path)

    print("Fetching Schaefer-200 atlas path...")
    atlas = fetch_atlas_schaefer_2018(
        n_rois=200,
        yeo_networks=7,
        resolution_mm=2
    )
    atlas_path = atlas.maps
    print("Atlas path:", atlas_path)

    bold_files = sorted(glob(os.path.join(
        root_dir,
        "sub-*",
        "ses-things*",
        "func",
        "*task-things*_bold.nii.gz"
    )))

    print(f"Found {len(bold_files)} image runs.")
    print("Using 5 workers...\n")

    with ProcessPoolExecutor(
        max_workers=5,
        initializer=init_worker,
        initargs=(mni_path, atlas_path)
    ) as executor:
        executor.map(process_bold, bold_files)

    print("\nImage 模态 ROI 提取完成！")
    print(f"Saved to: {save_root}")
