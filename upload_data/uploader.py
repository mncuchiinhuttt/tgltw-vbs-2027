import os
import sys
import time
import shutil
import zipfile
import requests
import subprocess
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, list_repo_files

# 1. Load env variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in the environment/.env file.", file=sys.stderr)
    sys.exit(1)

# 2. Setup Hugging Face API
api = HfApi()

try:
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info['name']
    print(f"Authenticated as Hugging Face user: {username}")
except Exception as e:
    print(f"Authentication failed. Please check your HF_TOKEN: {e}", file=sys.stderr)
    sys.exit(1)

REPO_ID = f"{username}/AIC2025"
print(f"Target repository ID: {REPO_ID}")

# 3. Create the repository if it doesn't exist (default to private=True for dataset security)
IS_PRIVATE = os.getenv("HF_REPO_PRIVATE", "true").lower() in {"1", "true", "yes"}
try:
    create_repo(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN, private=IS_PRIVATE, exist_ok=True)
    print(f"Dataset repository {REPO_ID} checked/created successfully (private={IS_PRIVATE}).")
except Exception as e:
    print(f"Failed to create/verify repository: {e}", file=sys.stderr)
    sys.exit(1)

# 4. Fetch already uploaded files to support skipping
try:
    print("Fetching already uploaded files from Hugging Face...")
    uploaded_files = set(list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN))
    print(f"Found {len(uploaded_files)} files in the repository.")
except Exception as e:
    print(f"Warning: Could not fetch repository files ({e}). Starting with empty set.")
    uploaded_files = set()

# 5. Define file lists and their target directories
sources = {
    "batch1.txt": "batch1",
    "batch2.txt": "batch2",
    "query.txt": "query"
}

# Collect all files to process
all_tasks = []
for file_list, target_dir in sources.items():
    if not os.path.exists(file_list):
        print(f"Warning: File list '{file_list}' not found.", file=sys.stderr)
        continue
    with open(file_list, 'r') as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename:
                continue
            zip_name_without_ext = filename[:-4] if filename.lower().endswith('.zip') else filename
            # The folder path inside repository: e.g. "batch1/Keyframes_L21"
            target_path_in_repo = f"{target_dir}/{zip_name_without_ext}"
            all_tasks.append({
                "url": url,
                "target_path_in_repo": target_path_in_repo,
                "filename": filename,
                "target_dir": target_dir,
                "zip_name_without_ext": zip_name_without_ext
            })

print(f"Total zip files in lists: {len(all_tasks)}")
todo_tasks = [t for t in all_tasks if f"{t['target_path_in_repo']}.done" not in uploaded_files]
print(f"Files already uploaded (detected via .done marker): {len(all_tasks) - len(todo_tasks)}")
print(f"Files to process: {len(todo_tasks)}")

# 6. Temporary directory setup
TEMP_DIR = "temp_download"
TEMP_UNZIP_DIR = "temp_unzip"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TEMP_UNZIP_DIR, exist_ok=True)

def extract_zip(zip_path, extract_base_dir, folder_name_default):
    """
    Extracts the zip file, and returns:
    (upload_source_dir, target_path_in_repo)
    """
    os.makedirs(extract_base_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        valid_members = [m for m in members if not m.startswith('__MACOSX') and not m.endswith('.DS_Store')]
        
        # Check if there is a common top-level directory
        top_dirs = {m.split('/')[0] for m in valid_members if '/' in m}
        has_top_dir = False
        if len(top_dirs) == 1:
            top_dir = list(top_dirs)[0]
            if all(m.startswith(top_dir + '/') or m == top_dir for m in valid_members):
                has_top_dir = True
                
        if has_top_dir:
            print(f"Zip contains a top-level directory '{top_dir}'. Extracting directly...")
            zip_ref.extractall(extract_base_dir)
            upload_source_dir = os.path.join(extract_base_dir, top_dir)
            target_path_in_repo = f"{os.path.basename(extract_base_dir)}/{top_dir}"
        else:
            print(f"Zip does not have a single top-level directory. Extracting to '{folder_name_default}'...")
            dest_dir = os.path.join(extract_base_dir, folder_name_default)
            os.makedirs(dest_dir, exist_ok=True)
            zip_ref.extractall(dest_dir)
            upload_source_dir = dest_dir
            target_path_in_repo = f"{os.path.basename(extract_base_dir)}/{folder_name_default}"
            
    return upload_source_dir, target_path_in_repo

# 7. Processing loop
for idx, task in enumerate(todo_tasks, 1):
    url = task['url']
    target_path_in_repo = task['target_path_in_repo']
    filename = task['filename']
    target_dir = task['target_dir']
    zip_name_without_ext = task['zip_name_without_ext']
    
    local_zip_path = os.path.join(TEMP_DIR, filename)
    local_unzip_base = os.path.join(TEMP_UNZIP_DIR, target_dir)

    print(f"\n[{idx}/{len(todo_tasks)}] Processing: {target_path_in_repo}")

    # 7.1 Download zip file with progress bar
    download_success = False
    try:
        local_file_exists = os.path.exists(local_zip_path)
        expected_size = None
        
        try:
            head_res = requests.head(url, timeout=30)
            if head_res.status_code == 200:
                expected_size = int(head_res.headers.get('content-length', 0))
        except Exception as e:
            print(f"Warning: HEAD request failed: {e}. Will perform regular download check.")

        if local_file_exists and expected_size and os.path.getsize(local_zip_path) == expected_size:
            print(f"Zip already fully downloaded locally: {local_zip_path} ({expected_size} bytes). Skipping download.")
            download_success = True
        else:
            print(f"Downloading {url} ...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_zip_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    size = f.write(data)
                    bar.update(size)
            download_success = True
            print(f"Downloaded successfully: {local_zip_path} ({os.path.getsize(local_zip_path)} bytes)")
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        if os.path.exists(local_zip_path):
            os.remove(local_zip_path)
        continue

    # 7.2 Unzip, upload folder, and clean up
    if download_success:
        extracted_dir = None
        try:
            # Unzip
            print(f"Unzipping {local_zip_path}...")
            extracted_dir, repo_upload_target = extract_zip(local_zip_path, local_unzip_base, zip_name_without_ext)
            print(f"Extracted to: {extracted_dir}")
            print(f"Repo upload target path: {repo_upload_target}")

            # Upload using hf upload CLI
            print(f"Uploading directory {extracted_dir} to HF as {repo_upload_target}...")
            cmd = [
                "hf", "upload",
                REPO_ID,
                extracted_dir,
                repo_upload_target,
                "--repo-type", "dataset",
                "--token", HF_TOKEN
            ]
            subprocess.run(cmd, check=True)
            print(f"Uploaded folder successfully to HF: {repo_upload_target}")

            # Upload .done marker to repo
            marker_filename = f"{zip_name_without_ext}.done"
            marker_local_path = os.path.join(TEMP_UNZIP_DIR, marker_filename)
            with open(marker_local_path, 'w') as f:
                f.write(f"Uploaded at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
            
            marker_repo_path = f"{target_dir}/{marker_filename}"
            print(f"Uploading done marker {marker_repo_path} to HF...")
            api.upload_file(
                path_or_fileobj=marker_local_path,
                path_in_repo=marker_repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            if os.path.exists(marker_local_path):
                os.remove(marker_local_path)
            print(f"Successfully uploaded done marker for {repo_upload_target}")

        except Exception as e:
            print(f"Error processing upload for {target_path_in_repo}: {e}", file=sys.stderr)
        finally:
            # Clean up local zip file
            if os.path.exists(local_zip_path):
                print(f"Cleaning up local zip: {local_zip_path}")
                os.remove(local_zip_path)
            # Clean up unzipped directory (we clean up the entire local_unzip_base/zip_name_without_ext or extracted_dir)
            if extracted_dir and os.path.exists(extracted_dir):
                print(f"Cleaning up extracted directory: {extracted_dir}")
                shutil.rmtree(extracted_dir)
            # Just in case, clean up local_unzip_base / folder_name_default if it is different
            default_dest = os.path.join(local_unzip_base, zip_name_without_ext)
            if os.path.exists(default_dest):
                shutil.rmtree(default_dest)

print("\nDone! All files processed.")