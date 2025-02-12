import subprocess

# 실행할 파일 목록
files_to_run = ['main_ensemble.py', 'main_masked_ensemble.py', 'main_mcdropout.py', 'main_zigzag.py', 'main_zigzagmy2.py']

for file in files_to_run:
    print(f"Running {file}...")
    subprocess.run(['python', file])  # 각 파일을 실행
    print(f"Finished {file}\n")