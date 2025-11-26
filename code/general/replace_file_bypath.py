import os
import shutil
from pathlib import Path

def replace_same_name_files(src_dir: str, dst_dir: str):
    """
    将 src_dir（文件夹B）中的文件，覆盖到 dst_dir（文件夹A）中同名的文件
    只替换同名文件，不删除 dst_dir 中多余的文件
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists() or not src_path.is_dir():
        print(f"错误：源文件夹不存在或不是文件夹 -> {src_path}")
        return
    if not dst_path.exists() or not dst_path.is_dir():
        print(f"错误：目标文件夹不存在或不是文件夹 -> {dst_path}")
        return

    replaced_count = 0
    skipped_count = 0

    # 遍历源文件夹 B 中的所有文件（不递归子文件夹，除非你需要）
    for src_file in src_path.iterdir():
        if src_file.is_file():               # 只处理文件，跳过子文件夹
            dst_file = dst_path / src_file.name

            if dst_file.exists():
                # 同名文件存在，直接覆盖
                shutil.copy2(src_file, dst_file)   # copy2 会保留元数据（时间戳等）
                print(f"已替换: {dst_file}")
                replaced_count += 1
            else:
                # 如果你希望把 B 中独有的文件也复制过去，把下面两行取消注释
                # shutil.copy2(src_file, dst_file)
                # print(f"已复制新文件: {dst_file}")
                print(f"跳过（A中无同名文件）: {src_file.name}")
                skipped_count += 1

    print("\n完成！")
    print(f"成功替换文件数: {replaced_count}")
    print(f"跳过文件数: {skipped_count}")

# ==============================
# 使用方法（直接修改下面两行路径即可）
# ==============================
if __name__ == "__main__":
    folder_A = r"E:\GEO\pyproject\casestudy\example"   # 目标文件夹（要被替换的）
    folder_B = r"E:\GEO\pyproject\casestudy\test"   # 源文件夹（用来替换的）
    replace_same_name_files(folder_B, folder_A)
    # Windows 示例：
    # folder_A = r"D:\项目\版本1