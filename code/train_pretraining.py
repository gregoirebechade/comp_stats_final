import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt


import torch.nn as nn





class Dataset_pretraining(torch.utils.data.Dataset):
    def __init__(self, path_to_data, n_files=36, n_samples_per_file=31*4, segment_length=1000, slide = 250):
        self.path_to_data = path_to_data
        self.n_files = n_files
        self.slide = slide
        self.n_samples_per_file = n_samples_per_file
        self.segment_length = segment_length
        self.data = []
        for file in range(self.n_files):
            # Précharger tous les fichiers en mémoire pour accélérer l'accès
            x = pd.read_csv(self.path_to_data + 's' + str(file).zfill(2) + '.csv', header=None).transpose().to_numpy()
            self.data.append(x)

    def __len__(self):
        return self.n_files * self.n_samples_per_file**2

    def __getitem__(self, idx):
        file = idx // self.n_samples_per_file**2
        sample = (idx % self.n_samples_per_file)
        first = (sample % (31*4))*250
        second = (sample // (31%4))*250
        if first + self.segment_length > len(self.data[file][0]):
            first = len(self.data[file][0]) - self.segment_length
        if second + self.segment_length > len(self.data[file][0]):
            second = len(self.data[file][0]) - self.segment_length
       
        x1 = self.data[file][:, first: first+self.segment_length]  # Utilisation de la donnée préchargée
        x2 = self.data[file][:, second: second+self.segment_length]
        # print(x1.shape)
        # print(x2.shape)
        # print(first, second)
        return torch.stack([torch.tensor(x1), torch.tensor(x2)]), torch.tensor([first, second])


class EEGFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=100):
        super(EEGFeatureExtractor, self).__init__()
        # input [batch_size, 19, 1000]
        self.conv1= nn.Conv1d(19, 32, 3, padding=1)
        self.conv2= nn.Conv1d(32, 64, 3, padding=1)
        self.conv3= nn.Conv1d(64, 128, 3, padding=1)
        self.conv4= nn.Conv1d(6, 10, 3, padding=1)   
        self.conv5 = nn.Conv1d(64, 15, 3, padding=1)
        self.pool = nn.MaxPool1d(1, 13)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(150, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print('au début', x.shape)
        x = self.relu(self.conv1(x))
        # print(1, x.shape)
        x = self.pool(x)
        # print(2, x.shape)
        x = self.relu(self.conv2(x))
        # print(3, x.shape)
        x = self.pool(x)
        # print(4, x.shape)
        x = torch.transpose(x, 1, 2)
        # print(5, x.shape)
        x = self.relu(self.conv4(x))
        # print(6, x.shape)
        x = torch.transpose(x, 1, 2)
        # print(7, x.shape)
        x = self.conv5(x)
        # print(8, x.shape)
        x = x.flatten(start_dim=1)
        # print(9, x.shape)
        x = self.dropout(x)
        # print('a la fin', x.shape)
        x = self.fc(x)
        return self.relu(x)


chemin_vers_sauvegarde = './../models/'
if __name__ == '__main__':
    dataloader_pretraining = DataLoader(Dataset_pretraining('./../data/kaggle_2/'), batch_size=1, shuffle=True)
    train_extractor = True
    tau = 516 # 1 seconde
    model_name='extractor'
    if not os.path.exists('./models/'+model_name):
        os.makedirs('./models/'+model_name)
    device = 'cpu'
    model = EEGFeatureExtractor()
    n_epochs=200
    loss = torch.nn.L1Loss()
    param_1 = torch.nn.Parameter(torch.ones(100, requires_grad=True))
    param_2 =  torch.nn.Parameter(torch.ones(1, requires_grad=True))
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': [param_1, param_2]}],
        lr=0.1
    )
    model.to(device)
    loss_train=[]
    if train_extractor:
        for epoch in (range(n_epochs)):
            print('epoch', epoch)
            losstrain=0
            counttrain=0
            lossval=0
            countval=0
            for batch_x,batch_y in dataloader_pretraining:
                batch_x=batch_x[0].to(device)
                batch_y = batch_y.long()
                batch_y=batch_y.to(device)
                optimizer.zero_grad()
                first_window = batch_x[0]
                second_window = batch_x[1]
                # print('the shape is', first_window.float().shape)
                first_prediction = model(first_window.float().unsqueeze(0))
                second_prediction = model(second_window.float().unsqueeze(0))
                label_predicted = torch.dot(param_1, abs(first_prediction - second_prediction).squeeze()) + param_2
                idx_1 = batch_y[0][0]
                idx_2 = batch_y[0][1]
                if (
                    
                    abs(idx_1- idx_2 ) < 1000 # close in time
                ) : 
                    y_pred = torch.tensor([-1]).to(device)
                else:
                    y_pred = torch.tensor([1]).to(device) # 1 s'ils sont proches, -1 sinon
                l= F.logsigmoid(y_pred * label_predicted)
                # l=torch.log(1+torch.exp(-y_pred*label_predicted))
                counttrain+=1
                l.backward()
                losstrain+=l
                optimizer.step()
            if epoch%10==0:
                print(f'epoch {epoch}, training loss = {losstrain/counttrain}')
            loss_train.append(losstrain/counttrain)
            
        torch.save(model, chemin_vers_sauvegarde+'_final'+'.pth')


        # saving the losses in txt files : 
        loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]



        with open('./losses/loss_train_'+model_name+'.txt', 'w') as f :
            for elt in loss_list_train : 
                f.write(str(elt) + '\n')