import os
import csv

# 指定 SDF 文件所在的文件夹路径
sdf_folder = os.getcwd()  # ← 替换成你的目录
output_csv = 'docking_scores.csv'

# 用于保存结果
results = []

# 遍历文件夹中的所有 .sdf 文件
for filename in os.listdir(sdf_folder):
    if filename.endswith('.sdf'):
        file_path = os.path.join(sdf_folder, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('>  <docking_score>  (1) '):
                    # 下一行就是分数值
                    if i + 1 < len(lines):
                        score = lines[i + 1].strip()
                        results.append((filename, score))
                    break  # 只取第一个 docking_score

# 写入 CSV 文件
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File', 'Docking_Score'])
    writer.writerows(results)

print(f"成功提取 {len(results)} 个分子得分，保存到 {output_csv}")
