#!/bin/bash

# 原始 sdf 文件所在目录
input_dir="./dock_result1_30_test"

# 输出文件夹的根目录
output_root="./dock_result1_30_test"

# 创建根目录
mkdir -p "$output_root"

# 遍历 sdf 文件
for sdf_file in "$input_dir"/*.sdf; do
    # 获取文件名（不带扩展名），例如 mol1
    filename=$(basename "$sdf_file" .sdf)

    # 构造新目录名和 pdb 文件名，例如 mol1_1
    new_name="${filename}_1"
    new_path="$output_root/$new_name"
    mkdir -p "$new_path"

    # 构造目标 pdb 路径：mol1_1/mol1_1.pdb
    pdb_file="$new_path/${new_name}.pdb"

    # 转换 sdf → pdb
    obabel "$sdf_file" -O "$pdb_file"
    echo "✅ Converted $filename.sdf → $pdb_file"
done
