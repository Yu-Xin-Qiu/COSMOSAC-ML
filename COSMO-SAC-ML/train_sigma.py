

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import yaml
import numpy as np
import torch
import time
import math
import pandas as pd
from model import ILBERT_sigma
from dataset import SMILES_sigma_dataset
import random
from torch.utils.data import Subset
from sklearn.model_selection import KFold
import torch




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def train(model, loader, optimizer):
    model.train()
    loss_all = []
    all_predictions = []
    all_targets = []

    with torch.set_grad_enabled(True):
        for datas, label, _ in loader:
            optimizer.zero_grad()

            data = [data.to(device) for data in datas]
            label = label.to(device)

            output = model(data)
            loss = F.l1_loss(output, label)

            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            all_predictions.append(output.detach().cpu())
            all_targets.append(label.detach().cpu())

    all_predictions = torch.cat([pred for pred in all_predictions if pred.dim() > 0], dim=0)
    all_targets = torch.cat([target for target in all_targets if target.dim() > 0], dim=0)

    mae_all = F.l1_loss(all_predictions, all_targets).item()
    r2_all = r2_score(all_predictions.flatten().numpy(), all_targets.flatten().numpy())

    mse_all_columns = []
    num_features = all_predictions.size(1)

    for i in range(num_features):
        mse_column = F.mse_loss(all_predictions[:, i], all_targets[:, i]).item()
        mse_all_columns.append(mse_column)
    # print(mse_all_columns)
    mse = np.mean(mse_all_columns)

    loss = np.average(loss_all)
    mae = mae_all
    r2 = r2_all

    return loss, mae, r2, mse





def evaluate_dataset(model, dataset, fold,shuffle,name, save=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

    y, pred,smi = [],[],[]


    with torch.no_grad():
        for datas, label,smiles in loader:

            data = [data.to(device) for data in datas]
            label = label.to(device)

            output = model(data)

            y.extend(label.detach().cpu().numpy())
            pred.extend(output.detach().cpu().numpy())
            smi.extend(smiles)


    y = torch.tensor(y)
    pred = torch.tensor(pred)

    mae_all_columns = []
    mse_all_columns = []

    num_features = pred.size(1)

    for i in range(num_features):
        mae_column = F.l1_loss(y[:, i], pred[:, i]).item()
        mse_column = F.mse_loss(y[:, i], pred[:, i]).item()

        mae_all_columns.append(mae_column)
        mse_all_columns.append(mse_column)


    r2 = r2_score(y.flatten(),pred.flatten())

    mae = np.mean(mae_all_columns)
    mse = np.mean(mse_all_columns)
    rmse = math.sqrt(mse)
    if save:
        print(len(y), len(pred), len(smi))

        y_df = pd.DataFrame(y.numpy())  
        pred_df = pd.DataFrame(pred.numpy()) 

        # 添加 SMILES 列到 y_df 和 pred_df
        y_df['SMILES'] = smi
        pred_df['SMILES'] = smi

        # 保存两个 DataFrame 为 CSV 文件
        y_df.to_csv(f'training_results/sigma/fold_{fold + 1}_experimental_{name}.csv', index=False)
        pred_df.to_csv(f'training_results/sigma/fold_{fold + 1}_predictions_{name}.csv', index=False)

    return mae, rmse, r2, mse





def cross_validation(dataset,config,df,num_folds):

    results = {'time': [], 'train_loss': [], 'train_r2': [], 'valid_mae': [], 'valid_rmse': [], 'valid_r2': [], 'test_mae' : [], 'test_mse': [], 'test_r2': []}
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
    indices = df[config['split by']].unique()

    for fold, (train_index, test_index) in enumerate(kf.split(indices)):

        train_indices = indices[train_index]
        test_indices = indices[test_index]
        # print(len(train_indices),len(test_index))


        train_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in train_indices])
        test_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in test_indices])


        print("Fold:", fold + 1)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))


        model = ILBERT_sigma(**config["transformer"]).to(device)

        state_dict = torch.load("model_weight/sigma/pretrained_sigma_model.pth", map_location=device)
        model.load_state_dict(state_dict, strict=False)


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_decay_patience'],
                                                                factor=config['lr_decay_factor'], min_lr=config['min_lr'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)


        best_valid_mae = float("inf")
        best_train_loss = float('inf')
        best_train_r2=0
        best_valid_mse=float('inf')
        best_valid_r2=0
        early_stopping_count = 0
        epoch_times = []

        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()

            model.train()
            train_loss, _, train_r2, _ = train(model, train_loader, optimizer)


            model.eval()

            valid_mae, _,valid_r2, valid_mse = evaluate_dataset(model,test_dataset ,fold,name='val',shuffle=False)


            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            print(f"Epoch{epoch:3d},Time:{epoch_time:.2f}s,TrainLoss:{train_loss:.6f},TrainR2:{train_r2:.6f},ValidMAE:{valid_mae:.6f},ValidMSE:{valid_mse:.6f},ValidR2:{valid_r2:.6f}")


            if train_r2 > 0 :
                if valid_mae < best_valid_mae :
                    best_train_loss = train_loss
                    best_train_r2 = train_r2
                    best_valid_mae = valid_mae
                    best_valid_mse = valid_mse
                    best_valid_r2 = valid_r2
                    torch.save(model.state_dict(), f'model_weight/sigma/fold_{fold + 1}_best_model.pth')

                    early_stopping_count = 0
                else:
                    early_stopping_count += 1

            else:
                pass

            if early_stopping_count >= config["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            lr_scheduler.step(valid_mae)

            current_lr = lr_scheduler.get_last_lr()
            print(f'Epoch {epoch}: Learning rate = {current_lr}')

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        model.load_state_dict(torch.load(f'model_weight/sigma/fold_{fold+1}_best_model.pth'))
        test_mae, _, test_r2,test_mse= evaluate_dataset(model, test_dataset, fold,shuffle=False,name='test',save=True)

        results['test_mae'].append(test_mae)
        results['test_mse'].append(test_mse)
        results['test_r2'].append(test_r2)

        print(f'Best_Epoch:{epoch - early_stopping_count}, Average epoch time: {average_epoch_time:.2f}s,Train_Loss: {best_train_loss:.6f}, Train_R2: {best_train_r2:.6f},Valid_MAE: {best_valid_mae:.6f}, Valid_MSE: {best_valid_mse:.6f}, Valid_R2: {best_valid_r2:.6f},Test_MAE: {test_mae:.6f}, Test_MSE: {test_mse:.6f}, Test_R2: {test_r2:.6f}')



    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for i in range(1, num_folds+1):
        filename1 = f"training_results/sigma/fold_{i}_predictions_test.csv"
        fold_df1 = pd.read_csv(filename1)
        df1 = pd.concat([df1, fold_df1])
        filename2 = f"training_results/sigma/fold_{i}_experimental_test.csv"
        fold_df2 = pd.read_csv(filename2)
        df2 = pd.concat([df2, fold_df2])



    y = torch.tensor(df1.drop(columns=['SMILES']).values)
    pred = torch.tensor(df2.drop(columns=['SMILES']).values)

    # 计算每列的 MAE, R2, MSE
    mae_all_columns = []
    mse_all_columns = []

    num_features = pred.size(1)

    for i in range(num_features):
        # 每列计算 MAE, R2, MSE
        mae_column = F.l1_loss(y[:, i], pred[:, i]).item()
        mse_column = F.mse_loss(y[:, i], pred[:, i]).item()

        mae_all_columns.append(mae_column)
        mse_all_columns.append(mse_column)

    r2 = r2_score(y.flatten(),pred.flatten())

    mae = np.mean(mae_all_columns)
    mse = np.mean(mse_all_columns)



    print('/n')
    print(mae,mse,r2)
    return r2





if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    setup_seed(config['seed'])
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    print('---loading dataset---')

    df = pd.read_csv("data/sigma_3423.csv")

    from tokenizer import SMILES_Atomwise_Tokenizer
    tokenizer=SMILES_Atomwise_Tokenizer('vocab.txt')
    dataset = SMILES_sigma_dataset(df = df, tokenizer = tokenizer)

    cross_validation(dataset, config, df,config['k'])

