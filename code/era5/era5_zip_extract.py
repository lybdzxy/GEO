import os
import zipfile
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

zip_dir = r"F:\ERA5\month\sfc\new"
out_dir = r"F:\ERA5\month\sfc\fin"
os.makedirs(out_dir, exist_ok=True)

def extract_and_rename(zip_file):
    zip_path = os.path.join(zip_dir, zip_file)
    base_name = os.path.splitext(zip_file)[0]  # ERA5_00z_sfc_194609
    extracted_files = []

    # 定义映射规则（长的优先，避免 avg 抢先匹配）
    mapping = {
        "data_stream-moda_stepType-avgad": "_moda_avgad.nc",
        "data_stream-moda_stepType-avgid": "_moda_avgid.nc",
        "data_stream-moda_stepType-avgua": "_moda_avgua.nc",
        "data_stream-moda_stepType-avg": "_moda_avg.nc",
        "data_stream-wamd_stepType-avgua": "_wamd_avgua.nc",
    }

    # 每个任务创建一个独立的临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        for member in os.listdir(tmpdir):
            if member.endswith(".nc"):
                src_path = os.path.join(tmpdir, member)

                # 默认情况：用原始 member 防止覆盖
                new_name = f"{base_name}_{member}"

                # 尝试匹配映射
                for key, suffix in mapping.items():
                    if key in member:
                        new_name = f"{base_name}{suffix}"
                        break

                dst_path = os.path.join(out_dir, new_name)

                # 用 copy 替代 replace，避免跨盘出错
                shutil.copy2(src_path, dst_path)
                extracted_files.append(dst_path)

    return f"{zip_file} -> {extracted_files}"

if __name__ == "__main__":
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(extract_and_rename, z): z for z in zip_files}

        for future in as_completed(futures):
            try:
                result = future.result()
                print("完成:", result)
            except Exception as e:
                print("出错:", futures[future], e)
