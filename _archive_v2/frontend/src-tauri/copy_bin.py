import shutil
import platform
import os

src = '/Users/ameliomar/Documents/company-brain/backend/dist/company_brain_backend'
dest_dir = '/Users/ameliomar/Documents/company-brain/frontend/src-tauri/binaries'
os.makedirs(dest_dir, exist_ok=True)

arch = platform.machine()
if arch == 'arm64':
    target = 'aarch64-apple-darwin'
else:
    target = 'x86_64-apple-darwin'

dest = os.path.join(dest_dir, f'company_brain_backend-{target}')
shutil.copy(src, dest)
print(f"Copied to {dest}")
