import subprocess
import os
import sys

cmd = 'source "$HOME/.cargo/env" && npx tauri build'
try:
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd='/Users/ameliomar/Documents/company-brain/frontend')
    with open('/Users/ameliomar/Documents/company-brain/frontend/build_output.txt', 'w') as f:
        f.write(result.stdout)
except subprocess.CalledProcessError as e:
    with open('/Users/ameliomar/Documents/company-brain/frontend/build_output.txt', 'w') as f:
        f.write(e.stdout)
