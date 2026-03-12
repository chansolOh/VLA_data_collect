import os
from pathlib import Path

# 🔧 폴더 경로
folder = Path("/nas/Dataset/VLA/UON/Isaacsim_OMY_apple_picking/action")   # ← 여기 폴더 경로로 변경

# 1️⃣ json 파일만 가져오기 + 이름순 정렬
files = sorted([f for f in folder.glob("*.json") if f.is_file()])

# 2️⃣ 먼저 임시 이름으로 변경 (이름 충돌 방지)
temp_files = []
for i, f in enumerate(files):
    tmp_name = folder / f"__tmp_{i:06d}.json"
    f.rename(tmp_name)
    temp_files.append(tmp_name)

# 3️⃣ 최종 이름으로 변경 (0000.json, 0001.json ...)
for i, f in enumerate(sorted(temp_files)):
    new_name = folder / f"{i:04d}.json"
    f.rename(new_name)

print("완료")
