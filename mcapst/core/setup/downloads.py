
import os
import gdown
from zipfile import ZipFile


# TODO: make a new drive with checkpoints and some additional, optional style templates as well

DEFAULT_CHECKPOINT_DIR = "checkpoints"


def _unzip_file(file_path: os.PathLike, destination: os.PathLike, delete_after=True) -> None:
    """ unzips a file into a subdirectory with the same name as the zip file (without extension) """
    with ZipFile(file_path, 'r') as zipped_file:
        zipped_file.extractall(destination)
    if delete_after:
        os.remove(file_path)  # remove the zip file after extraction

def download_google_drive(
    data_source: str,
    data_destination: os.PathLike = DEFAULT_CHECKPOINT_DIR
) -> None:
    """ Downloads a directory from a Google Drive folder and unzips it if necessary """
    # ensure the destination directory exists
    out_dir = os.path.dirname(data_destination)
    os.makedirs(out_dir, exist_ok=True)
    existing_files = set(os.listdir(out_dir))
    # attempt to download the content of the Google Drive folder and raise exception if it fails
    try:
        downloaded_files = gdown.download(data_source, data_destination, quiet=False, use_cookies=False)
        if not downloaded_files:
            raise Exception("No files were downloaded from the provided data source.")
        print(f"Downloaded files: {downloaded_files}")
    except Exception as e:
        raise Exception(f"An error occurred while downloading the dataset: {str(e)}")
    # check each downloaded file/folder and unzip if it's a zip file
    #? NOTE: could use downloaded_files instead of os.listdir, but I don't totally trust the gdown output to be consistent
    new_files: set[str] = set(os.listdir(out_dir)) - existing_files
    for file_name in new_files:
        file_path = os.path.join(out_dir, file_name)
        if file_name.endswith(('.zip', '.7z')):
            print(f"Found zip file: {file_name}. Unzipping...")
            _unzip_file(file_path, out_dir)
        else:
            print(f"Found non-zip file: {file_name}. Keeping it in the destination directory.")


def prompt_to_download_checkpoint(ckpt_path: str, drive_id: str):
    """ Prompts the user to download a checkpoint if it does not exist """
    from mcapst.core.utils.utils import get_user_confirmation
    user_prompt = f"CRITICAL: Checkpoint file '{ckpt_path}' not found but is required for this use case. Download the default checkpoint now?"
    if get_user_confirmation(f"\x1b[33m{user_prompt}\x1b[0m"):
        drive_link = f"https://drive.google.com/uc?export=download&id={drive_id}"
        download_google_drive(drive_link, ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' does not exist. Please provide a valid path.")