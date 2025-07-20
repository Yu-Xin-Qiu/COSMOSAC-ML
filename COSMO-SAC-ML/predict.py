import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from rdkit import Chem
from model import ILBERT_sigma, ILBERT
from dataset import SMILES_sigma_dataset, SMILES_dataset
from tokenizer import SMILES_Atomwise_Tokenizer

# 读取配置文件
config = yaml.load(open("config.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = SMILES_Atomwise_Tokenizer('vocab.txt')

# 预测sigma
def predict_sigma(df):
    
    dataset = SMILES_sigma_dataset(df=df, tokenizer=tokenizer)
    
    model = ILBERT_sigma(**config["transformer"]).to(device)
    model.load_state_dict(torch.load('model_weight/sigma/fold_0_best_model.pth'))
    model.eval()

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    pred_sigma = []

    with torch.no_grad():
        for datas, label, _ in loader:
            data = [data.to(device) for data in datas]
            label = label.to(device)
            output = model(data)

            pred_sigma.extend(output.detach().cpu().numpy())

            break

    pred_sigma = torch.tensor(pred_sigma)
    
    return pred_sigma

# 预测vcosmo
def predict_vcosmo(df):

    dataset = SMILES_dataset(df=df, tokenizer=tokenizer)
    
    model = ILBERT(**config["transformer"]).to(device)
    model.load_state_dict(torch.load('model_weight/vcosmo/fold_0_best_model.pth'))
    model.eval()

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    pred_vcosmo, smi = [], []

    with torch.no_grad():
        for datas, label, _ in loader:
            data = [data.to(device) for data in datas]
            label = label.to(device)
            output = model(data)
            output = output * 98.0288 + 207.1172  
            # tensor(207.1172) tensor(98.0288) torch.Size([12844])
            pred_vcosmo.extend(output.detach().cpu().numpy())

            break
    pred_vcosmo = np.array(pred_vcosmo).flatten()

    return pred_vcosmo

def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True,canonical=True)
    else:
        return None



def predict_sigma_vcosmo(smiles,draw=False):
    try:
        smiles = normalize_smiles(smiles)
    except Exception as e:
        print(f"Error normalizing SMILES: {e}")
        return

    # 初始化DataFrame
    column_names = [f"sigma_{i+1}" for i in range(51)] + ["Vcosmo"]
    data = {col: [0] for col in column_names}
    data["Normalized SMILES"] = [smiles]
    df = pd.DataFrame(data)
    
    # 获取sigma预测结果
    pred_sigma,pred_vcosmo = predict_sigma(df), predict_vcosmo(df)

    pred_sigma = np.maximum(pred_sigma, 0)  # 将负数替换为0

    # 将sigma的51列展平到 DataFrame 中，每一列的名称为 sigma_1 到 sigma_51
    sigma_columns = [f'sigma_{i+1}' for i in range(pred_sigma.shape[1])]
    pred_sigma_df = pd.DataFrame(pred_sigma.numpy(), columns=sigma_columns)

    if draw:
        # 画图
        x = np.linspace(-0.025, 0.025, num=51)  # 横坐标：-0.025到0.025，间隔0.001
        y = pred_sigma_df.iloc[0].values  # 获取第一行的sigma值作为纵坐标

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f"Pred_vcosmo = {pred_vcosmo[0]:.2f}", marker='o', color='b')

        plt.xlabel('Sigma Values')
        plt.ylabel('Predicted Sigma')
        plt.title(f"Prediction for SMILES: {smiles}")
        plt.legend()
        plt.grid(True)
        plt.show()
        return pred_sigma_df, pred_vcosmo
    else:
        return pred_sigma_df, pred_vcosmo



# predict_sigma_vcosmo("O",draw=True)  
