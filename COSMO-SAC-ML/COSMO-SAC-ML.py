

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import yaml
import numpy as np
import torch
import time
import math
import pandas as pd
from model import ILBERT_T_AC
from dataset import SMILES_dataset_IDAC
import random
from torch.utils.data import Subset
from sklearn.model_selection import KFold


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
        for datas,label,_,_,_,_,_ in loader:
            optimizer.zero_grad()
            data = [data.to(device) for data in datas]
            label = label.to(device)
            label = normalizer.norm(label)
            output = model(data)

            label = torch.squeeze(normalizer.denorm(label))
            output = torch.squeeze(normalizer.denorm(output))

            loss = F.mse_loss(output, label)

            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            all_predictions.append(output.detach().cpu())
            all_targets.append(label.detach().cpu())

    all_predictions = torch.cat([pred for pred in all_predictions if pred.dim() > 0], dim=0)
    all_targets = torch.cat([target for target in all_targets if target.dim() > 0], dim=0)

    mae_all = F.l1_loss(all_predictions, all_targets).item()
    r2_all = r2_score(all_predictions.numpy(), all_targets.numpy())
    mse_all = F.mse_loss(all_predictions, all_targets).item()

    loss = np.average(loss_all)
    mae = mae_all
    r2 = r2_all
    rmse = math.sqrt(mse_all)

    return loss, mae, r2, rmse

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def evaluate_dataset(model, dataset, fold,shuffle,name, save=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

    y, pred,smi1,smi2,T,IDAC,IDAC_COSMO = [],[],[],[],[],[],[]


    with torch.no_grad():
        for datas, label,smiles1,smiles2,t,idac,idac_COSMO in loader:

            data = [data.to(device) for data in datas]
            label = label.to(device)
            label = normalizer.norm(label)
            output = model(data)

            label = normalizer.denorm(label)
            output = normalizer.denorm(output)

            y.extend(label.detach().cpu().numpy())
            pred.extend(output.detach().cpu().numpy())
            smi1.extend(smiles1)
            smi2.extend(smiles2)
            T.extend(t.detach().cpu().numpy())
            IDAC.extend(idac.detach().cpu().numpy())
            IDAC_COSMO.extend(idac_COSMO.detach().cpu().numpy())


    y, pred= np.array(y).flatten(), np.array(pred).flatten()
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred)

    if save:
        df = pd.DataFrame({'Experimental Value': y, 'Predicted Value': pred, 'SMILES1': smi1,'SMILES2': smi2, 'T': T,'IDAC':IDAC,'IDAC_COSMO':IDAC_COSMO})
        df.to_csv(f'training_results/ML/fold_{fold + 1}_predictions_{name}.csv', index=False)

    return mae, rmse, r2, mse


def cross_validation(dataset,config,df,num_folds):

    results = {'time': [], 'train_loss': [], 'train_r2': [], 'valid_mae': [], 'valid_rmse': [], 'valid_r2': [],'test_mae': [], 'test_rmse': [], 'test_r2': []}
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
    indices = df[config['split by']].unique()

    for fold, (train_index, test_index) in enumerate(kf.split(indices)):

        train_indices = indices[train_index]
        test_indices = indices[test_index]
        # print(len(train_indices),len(test_index))


        train_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in train_indices])
        valid_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in test_indices])


        print("Fold:", fold + 1)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(valid_dataset))


        model = ILBERT_T_AC(**config["transformer"]).to(device)


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_decay_patience'],
                                                                factor=config['lr_decay_factor'], min_lr=config['min_lr'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)


        best_valid_mae = float("inf")
        best_train_loss = float('inf')
        best_train_r2=0
        best_valid_rmse=float('inf')
        best_valid_r2=0
        early_stopping_count = 0
        epoch_times = []

        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()

            model.train()
            train_loss, _, train_r2, _ = train(model, train_loader, optimizer)


            model.eval()
            valid_mae, valid_rmse, valid_r2, _ = evaluate_dataset(model,valid_dataset ,fold,name='val',shuffle=False,save=True)

            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            print(f"Epoch{epoch:3d},Time:{epoch_time:.2f}s,TrainLoss:{train_loss:.6f},TrainR2:{train_r2:.6f},ValidMAE:{valid_mae:.6f},ValidRMSE:{valid_rmse:.6f},ValidR2:{valid_r2:.6f}")



            if valid_rmse < best_valid_rmse :
                best_train_loss = train_loss
                best_train_r2 = train_r2
                best_valid_mae = valid_mae
                best_valid_rmse = valid_rmse
                best_valid_r2 = valid_r2
                torch.save(model.state_dict(), f'model_weight/ML/fold_{fold + 1}_best_model.pth')

                early_stopping_count = 0
            else:
                early_stopping_count += 1


            if early_stopping_count >= config["early_stop_patience"]:
                print(f"/nEarly stopping at epoch {epoch + 1}")
                break

            lr_scheduler.step(valid_rmse)

            current_lr = lr_scheduler.get_last_lr()
            print(f'Epoch {epoch}: Learning rate = {current_lr}')

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        model.load_state_dict(torch.load(f'model_weight/ML/fold_{fold+1}_best_model.pth'))
        test_mae, test_rmse, test_r2,_= evaluate_dataset(model, valid_dataset, fold,shuffle=False,name='test',save=True)

        results['test_mae'].append(test_mae)
        results['test_rmse'].append(test_rmse)
        results['test_r2'].append(test_r2)



        print(f'Best_Epoch:{epoch - early_stopping_count}, Average epoch time: {average_epoch_time:.2f}s,Train_Loss: {best_train_loss:.6f}, Train_R2: {best_train_r2:.6f},Valid_MAE: {best_valid_mae:.6f}, Valid_RMSE: {best_valid_rmse:.6f}, Valid_R2: {best_valid_r2:.6f},Test_MAE: {test_mae:.6f}, Test_RMSE: {test_rmse:.6f}, Test_R2: {test_r2:.6f}')




    df = pd.DataFrame()
    for i in range(1, num_folds+1):
        filename = f"training_results/ML/fold_{i}_predictions_test.csv"
        fold_df = pd.read_csv(filename)
        df = pd.concat([df, fold_df])

    mae = mean_absolute_error(df['Experimental Value'], df['Predicted Value'])
    rmse = np.sqrt(mean_squared_error(df['Experimental Value'], df['Predicted Value']))
    R2 = r2_score(df['Experimental Value'], df['Predicted Value'])

    print('/n')
    print(mae,rmse,R2)

    return R2







if __name__ == "__main__":
    config = yaml.load(open("config_ML.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

    setup_seed(config['seed'])
    print(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    print('---loading dataset---')

    df_train = pd.read_csv("data/IDAC.csv")

    from tokenizer import SMILES_Atomwise_Tokenizer
    tokenizer = SMILES_Atomwise_Tokenizer('merged_output.txt')
    dataset = SMILES_dataset_IDAC(df = df_train, tokenizer = tokenizer)

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    labels = []
    for data,label,_,_,_,_,_ in loader:
        labels.append(label)
    labels = torch.cat(labels)
    normalizer = Normalizer(labels)
    print(normalizer.mean, normalizer.std, labels.shape)

    cross_validation(dataset, config, df_train,config['k'])





