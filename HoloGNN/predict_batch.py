
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import pandas as pd
import torch
from HoloGNN import HoloGNN
from dataset_GIGN import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
import pandas as pd

# %%
def val(model, dataloader, device):
    model.eval()

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y

            print(f'pred is {pred}')
            print(f'len pred {len(pred)}')
            print(f'label is {label}')
            print(f'len pred {len(label)}')

            pred_np = pred.view(-1).cpu().numpy()
            label_np = label.view(-1).cpu().numpy()

            df = pd.DataFrame({
                'label': label_np,
                'prediction': pred_np
            })

            df.to_csv('label_pred.csv', index=False)

            assert len(pred) == len(label)
            # Step 1️⃣: 找到 label==25 对应的 pred 值
            mask_25 = label == 25
            pred_25 = pred[mask_25]
            print(f"🔍 Prediction where label == 25: {pred_25.item():.4f}")

            # Step 2️⃣: 按 pred 从大到小排序，获取排序后的索引
            sorted_indices = torch.argsort(pred, descending=True)

            # 获取排序后的 pred 和 label
            pred_sorted = pred[sorted_indices]
            label_sorted = label[sorted_indices]

            # 打印前几项示例
            print("\n📊 Top sorted predictions:")
            for i in range(len(pred)):  
                print(f"Label: {int(label_sorted[i].item()):2d}, Pred: {pred_sorted[i].item():.4f}")

    
# %%
import argparse
data_root = './data'
graph_type = 'Graph_HoloGNN'
batch_size = 64

parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('--docking_dir', type=str, required=True, help='the result of docking')
parser.add_argument('--csv_file', type=str, required=True, help='the index of molecule')
args = parser.parse_args()

docking_dir = args.docking_dir
csv_file = args.csv_file

valid_dir = os.path.join(data_root, docking_dir)


valid_df = pd.read_csv(os.path.join(data_root, csv_file))


valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)


valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)


device = torch.device('cuda:0')
model = HoloGNN(35, 256).to(device)
load_model_dict(model, './model/20250604_114310_HoloGNN_repeat1/model/epoch-314, train_loss-0.1701, train_rmse-0.4125, valid_rmse-1.1381, valid_pr-0.7888.pt')
model = model.cuda()

val(model, valid_loader, device)


# %%
