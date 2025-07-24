import pandas as pd
import matplotlib.pyplot as plt

# 读取合并好的数据
df = pd.read_csv('merged_output.csv')

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(df['docking_score'], df['prediction'], color='royalblue', alpha=0.7)

# 找到 id 为 25 的行
highlight = df[df['id'] == 25]

# 如果存在该点，则绘制高亮点
if not highlight.empty:
    plt.scatter(highlight['docking_score'], highlight['prediction'],
                color='crimson', s=100, marker='*', label='ID = 25')
    # 可选：添加文本标签
    x = highlight['docking_score'].values[0]
    y = highlight['prediction'].values[0]
    plt.text(x, y + 0.1, 'ID 25', fontsize=10, color='crimson', ha='center')

# 添加标签和标题
plt.xlabel('Docking Score (kcal/mol)')
plt.ylabel('Predicted Affinity')
plt.ylim(3, 8)
plt.xlim(0, -6)
plt.title('Docking Score vs. Predicted Affinity')
plt.grid(True)

# 可选：保存图像
plt.savefig('docking_vs_prediction_1.png', dpi=300)

# 显示图像
plt.show()
