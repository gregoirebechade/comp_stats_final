
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from train_pretraining import EEGFeatureExtractor
from torch.utils.data import DataLoader
import os 
import numpy as np




class EEGClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(EEGClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(100, 1)
        self.f = nn.Sigmoid()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = F.normalize(features, p=2, dim=1)
        x = self.fc(features)
        return self.f(x)

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data # par example './../data/train/
        self.X = os.listdir(self.path_to_data) # the list of the files in the train set 
        self.data=[]
        self.labels=np.array([0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1])
        for idx in range(len(self.X)):
            x = pd.read_csv(self.path_to_data + self.X[idx], header=None).to_numpy()
            self.data.append((x, self.labels[int(self.X[idx].split('_')[0])]))


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x , y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)
        # labels=np.array([0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1])
        # x = pd.read_csv(self.path_to_data + self.X[idx], header=None).to_numpy()
        # record_number = int(self.X[idx].split('_')[0])
        # return torch.tensor(x), torch.tensor(labels[record_number]) # un eeg sur 1000 échantillonages et le label correspondant





if __name__ == '__main__':
    chemin_vers_sauvegarde = './models/'
    # Load data
    pretrained = torch.load('./models/extractor_final.pth')
    dataloader_train = DataLoader(Mydataset('./../data/train/'), batch_size=5, shuffle=True)
    device = 'cpu'
    for param in pretrained.parameters():
        param.requires_grad = False
    model = EEGClassifier(pretrained)
    # loss for classification : 
    # loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.L1Loss()
    model_name = 'classifier_pretrained'
    loss_train=[]
    n_epochs=200
    optimizer = torch.optim.Adam(model.fc.parameters())
    for epoch in range(n_epochs):
        print('epoch', epoch)
        losstrain=0
        counttrain=0
        for batch_x,batch_y in dataloader_train:
            batch_x=batch_x.to(device)
            batch_y = batch_y.float()
            batch_y=batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_x.float())
            l=loss(y_pred.squeeze(), batch_y)
            counttrain+=1
            l.backward()
            losstrain+=l
            optimizer.step()
        if epoch%10==0:
            print(f'epoch {epoch}, training loss = {losstrain/counttrain}')
        loss_train.append(losstrain/counttrain)
    torch.save(model, chemin_vers_sauvegarde+model_name+'_final_bis'+'.pth')
    loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]
    with open('./losses/loss_train_'+model_name+'.txt', 'w') as f :
        for elt in loss_list_train : 
            f.write(str(elt) + '\n')
    








    print('finsih model 1')
    print('begin model 2')
    not_pretrained = EEGFeatureExtractor()
    model = EEGClassifier(not_pretrained)
    loss = torch.nn.BCEWithLogitsLoss()
    n_epochs=200
    optimizer = torch.optim.Adam(model.parameters())
    model_name = 'classifier_not_pretrained'
    loss_train=[]
    # loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.L1Loss()
    n_epochs=200
    optimizer = torch.optim.Adam(model.fc.parameters())
    for epoch in range(n_epochs):
        print('epoch', epoch)
        losstrain=0
        counttrain=0
        for batch_x,batch_y in dataloader_train:
            batch_x=batch_x.to(device)
            batch_y = batch_y.float()
            batch_y=batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_x.float())
            l=loss(y_pred.squeeze(), batch_y)
            counttrain+=1
            l.backward()
            losstrain+=l
            optimizer.step()
        if epoch%10==0:
            print(f'epoch {epoch}, training loss = {losstrain/counttrain}')
        loss_train.append(losstrain/counttrain)
    torch.save(model, chemin_vers_sauvegarde+model_name+'_final'+'.pth')
    loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]
    with open('./losses/loss_train_'+model_name+'bis'+'.txt', 'w') as f :
        for elt in loss_list_train : 
            f.write(str(elt) + '\n')

