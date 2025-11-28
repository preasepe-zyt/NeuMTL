

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from model import NeuMTL


from tqdm import tqdm
import sys, os
import time
import pickle
import random
from gra import *

import os
import torch.distributed as dist

seed = 4221
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
  generator = torch.Generator('cuda').manual_seed(seed)
else:
  generator = torch.Generator().manual_seed(seed)



def train(model, device, train_loader1, train_loader2, train_loader3, train_loader4, train_loader5, optimizer, mse_f, bce_f, epoch, FLAGS):
    model.train()


    # 设置长的 dataloader 长度
    total_steps = max(len(train_loader1), len(train_loader2), len(train_loader3), len(train_loader4), len(train_loader5))

    iter_loader1 = iter(train_loader1)
    iter_loader2 = iter(train_loader2)
    iter_loader3 = iter(train_loader3)
    iter_loader4 = iter(train_loader4)
    iter_loader5 = iter(train_loader5)
    
    with tqdm(range(total_steps), desc=f"Epoch {epoch + 1}") as t:
        for i in t:
            try:
                data1 = next(iter_loader1)
            except StopIteration:
                iter_loader1 = iter(train_loader1)
                data1 = next(iter_loader1)
                
            try:
                data2 = next(iter_loader2)
            except StopIteration:
                iter_loader2 = iter(train_loader2)
                data2 = next(iter_loader2) 
                
            try:
                data3 = next(iter_loader3)
            except StopIteration:
                iter_loader3 = iter(train_loader3)
                data3 = next(iter_loader3) 
                
            try:
                data4 = next(iter_loader4)
            except StopIteration:
                iter_loader4 = iter(train_loader4)
                data4 = next(iter_loader4) 
                
            try:
                data5 = next(iter_loader5)
            except StopIteration:
                iter_loader5 = iter(train_loader5)
                data5 = next(iter_loader5) 


            optimizer.zero_grad()
            Pridection, classification1, classification2, classification3, classification4 = model(data1.to(device), data2.to(device), data3.to(device), data4.to(device), data5.to(device))

            mse_loss = mse_f(Pridection, data1.y.view(-1, 1).float().to(device))
            bce_loss1 = bce_f(classification1, data2.y.view(-1, 1).float().to(device))
            bce_loss2 = bce_f(classification2, data3.y.view(-1, 1).float().to(device))
            bce_loss3 = bce_f(classification3, data4.y.view(-1, 1).float().to(device))
            bce_loss4 = bce_f(classification4, data5.y.view(-1, 1).float().to(device))


            train_ci = get_cindex(Pridection.cpu().detach().numpy(), data1.y.view(-1, 1).float().cpu().detach().numpy())

            train_tasks = [mse_loss, bce_loss1, bce_loss2, bce_loss3, bce_loss4]
            
            rng = np.random.default_rng()
            grad_dims = []
            for mm in model.shared_modules():        
                 for param in mm.parameters():     
                     grad_dims.append(param.data.numel()) 
            grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)

            
            for i, loss_i in enumerate(train_tasks):
                    loss_i.backward(retain_graph=True)   
                    NeuGradBalancervec(model, grads, grad_dims, i) 
                    model.zero_grad_shared_modules()   
                    
            g = NeuGradBalancer(grads)          
            overwrite_grad(model, g, grad_dims, len(train_tasks)) 

            optimizer.step()
            t.set_postfix(MSE=mse_loss.item(), bce1=bce_loss1.item(), bce2=bce_loss2.item(), bce3=bce_loss3.item(), bce4=bce_loss4.item())
        msg = f"Epoch {epoch+1},  MSE={mse_loss.item()}, bce1={bce_loss1.item()}, bce2={bce_loss2.item()}, bce3={bce_loss3.item()}, bce4={bce_loss4.item()}"
        logging(msg, FLAGS)
    return model

def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def test(model, device, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5, mse_f, bce_f, FLAGS):
    """Test the NeuMTL model on the specified data and report the results.""" 
    print('Testing on {} samples...'.format(len(test_loader1.dataset),len(test_loader2.dataset),len(test_loader3.dataset),len(test_loader4.dataset),len(test_loader5.dataset)))
    model.eval()
    total_true1 = torch.Tensor()
    total_predict1 = torch.Tensor()
    
    total_true2 = torch.Tensor()
    total_predict2 = torch.Tensor()

    total_true3 = torch.Tensor()
    total_predict3 = torch.Tensor()

    total_true4 = torch.Tensor()
    total_predict4 = torch.Tensor()

    total_true5 = torch.Tensor()
    total_predict5 = torch.Tensor()


    total_loss = 0 

    if dataset == "kiba":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  

    loader1 = infinite_loader(test_loader1)
    loader2 = infinite_loader(test_loader2)
    loader3 = infinite_loader(test_loader3)
    loader4 = infinite_loader(test_loader4)
    loader5 = infinite_loader(test_loader5)

    with torch.no_grad():
       for _ in tqdm(range(max(len(test_loader1), len(test_loader2),len(test_loader3), len(test_loader4),len(test_loader5))), desc="Testing"):
            data1 = next(loader1)
            data2 = next(loader2)
            data3 = next(loader3)
            data4 = next(loader4)
            data5 = next(loader5)
            
            Pridection, classification1, classification2, classification3, classification4 = model(data1.to(device), data2.to(device), data3.to(device), data4.to(device), data5.to(device))
            
            #Pridection
            total_true1 = torch.cat((total_true1, data1.y.view(-1, 1).cpu()), 0)
            total_predict1 = torch.cat((total_predict1, Pridection.cpu()), 0)   
            G = total_true1.numpy().flatten()
            P = total_predict1.numpy().flatten()
            mse_loss = mse(G, P)
            test_ci = get_cindex(G, P)      
            rm2 = get_rm2(G, P)   
            auc_values = []
            for t in thresholds:
                auc = get_aupr(np.int32(G > t), P)
                auc_values.append(auc) 

            #classification
            total_true2 = torch.cat((total_true2, data2.y.view(-1, 1).cpu()), 0)
            total_predict2 = torch.cat((total_predict2, classification1.cpu()), 0) 
            perform1 = get_performace(total_true2, total_predict2)

            total_true3 = torch.cat((total_true3, data3.y.view(-1, 1).cpu()), 0)
            total_predict3 = torch.cat((total_predict3, classification2.cpu()), 0) 
            perform2 = get_performace(total_true3, total_predict3)

            total_true4 = torch.cat((total_true4, data4.y.view(-1, 1).cpu()), 0)
            total_predict4 = torch.cat((total_predict4, classification3.cpu()), 0) 
            perform3 = get_performace(total_true4, total_predict4)


            total_true5 = torch.cat((total_true5, data5.y.view(-1, 1).cpu()), 0)
            total_predict5 = torch.cat((total_predict5, classification4.cpu()), 0) 
            perform4 = get_performace(total_true5, total_predict5)

            mse_loss = mse_f(Pridection, data1.y.view(-1, 1).float().to(device))
            bce_loss1 = bce_f(classification1, data2.y.view(-1, 1).float().to(device))
            bce_loss2 = bce_f(classification2, data3.y.view(-1, 1).float().to(device))
            bce_loss3 = bce_f(classification3, data4.y.view(-1, 1).float().to(device))
            bce_loss4 = bce_f(classification4, data5.y.view(-1, 1).float().to(device))

            total_loss =  mse_loss + bce_loss1 + bce_loss2 + bce_loss3+ bce_loss4

            
    return  total_loss, mse_loss, test_ci, rm2, auc_values, G, P, perform1, perform2, perform3, perform4

def experiment(FLAGS, dataset1, dataset2, dataset3, dataset4, dataset5, device):
    logging('Starting program', FLAGS)


    # Hyperparameters
    BATCH_SIZE = 128
    LR = 0.0002
    NUM_EPOCHS = 100

    # Print hyperparameters
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Log hyperparameters
    msg = f"Dataset {dataset1, dataset2, dataset3, dataset4, dataset5}, Device {device}, batch size {BATCH_SIZE}, learning rate {LR}, epochs {NUM_EPOCHS}"
    logging(msg, FLAGS)

    # Load tokenizer
    with open(f'data/{dataset1}_tokenizer.pkl', 'rb') as f:
        tokenizer1 = pickle.load(f)

    with open(f'data2/{dataset2}_tokenizer.pkl', 'rb') as f:
        tokenizer2 = pickle.load(f)

    with open(f'data2/{dataset3}_tokenizer.pkl', 'rb') as f:
        tokenizer3 = pickle.load(f)

    
    with open(f'data2/{dataset4}_tokenizer.pkl', 'rb') as f:
        tokenizer4 = pickle.load(f)

    
    with open(f'data2/{dataset5}_tokenizer.pkl', 'rb') as f:
        tokenizer5 = pickle.load(f)

        
    # Load processed data
    processed_data_file_train = f"data/processed/{dataset}_train.pt"
    processed_data_file_test = f"data/processed/{dataset}_test.pt"
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
    else:
        train_data1 = TestbedDataset(root="data", dataset=f"{dataset1}_train")
        test_data1 = TestbedDataset(root="data", dataset=f"{dataset1}_test")

        train_data2 = TestbedDataset(root="data2", dataset=f"{dataset2}_train")
        test_data2 = TestbedDataset(root="data2", dataset=f"{dataset2}_test")

        
        train_data3 = TestbedDataset(root="data2", dataset=f"{dataset3}_train")
        test_data3 = TestbedDataset(root="data2", dataset=f"{dataset3}_test")

        
        train_data4 = TestbedDataset(root="data2", dataset=f"{dataset4}_train")
        test_data4 = TestbedDataset(root="data2", dataset=f"{dataset4}_test")

        
        train_data5 = TestbedDataset(root="data2", dataset=f"{dataset5}_train")
        test_data5 = TestbedDataset(root="data2", dataset=f"{dataset5}_test")

        # Prepare PyTorch mini-batches
        train_loader1 = DataLoader(train_data1, batch_size=BATCH_SIZE, shuffle=True)
        test_loader1 = DataLoader(test_data1, batch_size=BATCH_SIZE, shuffle=False)

        train_loader2 = DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True)
        test_loader2 = DataLoader(test_data2, batch_size=BATCH_SIZE, shuffle=False)

        train_loader3 = DataLoader(train_data3, batch_size=BATCH_SIZE, shuffle=True)
        test_loader3 = DataLoader(test_data3, batch_size=BATCH_SIZE, shuffle=False)

        train_loader4 = DataLoader(train_data4, batch_size=BATCH_SIZE, shuffle=True)
        test_loader4 = DataLoader(test_data4, batch_size=BATCH_SIZE, shuffle=False)

        train_loader5 = DataLoader(train_data5, batch_size=BATCH_SIZE, shuffle=True)
        test_loader5 = DataLoader(test_data5, batch_size=BATCH_SIZE, shuffle=False)


        model = NeuMTL(tokenizer1, device).to(device)
                

        optimizer = optim.Adam(model.parameters(), lr=LR) #FetterGrad(
        mse_f = nn.MSELoss()
        bce_f = nn.BCELoss()

        final_re = pd.DataFrame()
        final_cla = pd.DataFrame()
        best_loss = float('inf')  
        for epoch in range(NUM_EPOCHS):

            
            model = train(model, device, train_loader1, train_loader2, train_loader3, train_loader4, train_loader5, optimizer, mse_f, bce_f, epoch, FLAGS)

            if (epoch + 1) % 1 == 0:
                # Test model
                total_loss, mse_loss,test_ci, rm2, auc_values, G, P, perform1, perform2, perform3, perform4 = test(model, device, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5, mse_f, bce_f, FLAGS)
                filename = f"saved_models/NeuMTL_model_{dataset1, dataset2, dataset3, dataset4, dataset5}.pth"
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(model.state_dict(), filename)
                    print('model saved')
                # 指标名称列表
                metric_names = ["ROC-AUC", "PRC-AUC", "Accuracy", "Balanced Acc","MCC", "Kappa", "Sensitivity", "Specificity","Precision", "F1-score"]
                dat_name = ["metric","BBB", "NC", "NA", "NT","epoch"]
                # 四个模型的 performance 列表
                perform_list = [metric_names, perform1, perform2, perform3, perform4,len(perform4)*[int(epoch)]]
                cur_cla = pd.DataFrame(perform_list,index=dat_name).transpose()
                cur_re = pd.DataFrame({"MSE":mse_loss.item(),"CI":test_ci, "RM2":rm2,"AUC":sum(auc_values)/len(auc_values),"epoch":epoch}, index=["Result"])

                print(cur_cla)
                print(cur_re)

                final_re = pd.concat([cur_re,final_re],axis=0)
                final_cla = pd.concat([cur_cla,final_cla],axis=0)

                   
        final_re.to_csv(f"regression_{dataset1}_grad.csv", index=False)
        final_cla.to_csv(f"classification_{dataset1}_grad.csv", index=False)
        # Save estimated and true labels
        folder_path = "Affinities/"
        np.savetxt(folder_path + f"estimated_labels_{dataset}.txt", P)
        np.savetxt(folder_path + f"true_labels_{dataset}.txt", G)

        logging('Program finished', FLAGS)

if __name__ == "__main__":

    datasets = [ 'bindingdb', 'davis', 'kiba']
    dataset_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dataset = datasets[dataset_idx]

    dataset2 = ['BBB', "to_NC", "to_NA", "to_NT"]
    

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:" + str(int(sys.argv[2])) if len(sys.argv) > 2 and torch.cuda.is_available() else default_device)

    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = f'dataset_{dataset}_total_{int(time.time())}'


    os.makedirs(FLAGS.log_dir, exist_ok=True)
    os.makedirs('Affinities', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    experiment(FLAGS, dataset, dataset2[0], dataset2[1], dataset2[2], dataset2[3], device)
    
