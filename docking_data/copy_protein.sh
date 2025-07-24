#!/bin/bash

# 蛋白质文件路径
protein_file="./6o0h_protein.pdb"

# 目标文件夹根路径（你可以根据实际修改）
target_root="./dock_result1_30_test"

# 遍历所有以 *_1 结尾的目录
for folder in "$target_root"/*_1/; do
    if [ -d "$folder" ]; then
        cp "$protein_file" "$folder"
        echo "✅ Copied protein.pdb to $folder"
    fi
done
