from pathlib import Path
from huggingface_hub import login, upload_folder

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "rbc"

login()
for subdir in ["2D", "3D"]:
    upload_folder(
        folder_path=str(DATA_DIR / subdir),
        path_in_repo=subdir,
        repo_id="tmarkmann/dataset-rbc-fno",
        repo_type="dataset",
    )
