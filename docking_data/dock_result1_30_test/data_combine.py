import csv

# 读取 docking_scores.csv
docking_dict = {}
with open('docking_scores.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['File']
        if filename.startswith('mol') and filename.endswith('.sdf'):
            mol_id = int(filename[3:-4])  # 从 mol25.sdf 提取出 25
            docking_dict[mol_id] = row['Docking_Score']

# 读取 label.csv
label_dict = {}
with open('label_pred.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_id = int(float(row['label']))  # 处理如 25.0 → 25
        label_dict[label_id] = row['prediction']

# 取交集
common_ids = sorted(set(docking_dict.keys()) & set(label_dict.keys()))

# 写入新文件
with open('merged_output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'docking_score', 'prediction'])
    for id_ in common_ids:
        writer.writerow([id_, docking_dict[id_], label_dict[id_]])

print(f"成功合并 {len(common_ids)} 条记录，保存为 merged_output.csv")
