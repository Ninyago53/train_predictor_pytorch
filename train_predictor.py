import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



# load the training data 

x_coco = np.load("clip-retrieval/embs_localized_coco/text_emb/text_emb_0.npy")
x_laion = np.load("clip-retrieval/embs_laion/text_emb/text_emb_0.npy")
x = np.vstack([x_coco, x_laion])


y = np.zeros(x.shape[0])
for i in range(x_coco.shape[0]):
    y[i] = 1

y = np.random.permutation(y)

val_percentage = 0.05 # 5% of the trainingdata will be used for validation

shape = x.shape[0]
train_border = int(shape * (1 - val_percentage) )

train_tensor_x = torch.Tensor(x[:train_border]) # transform to torch tensor
train_tensor_y = torch.Tensor(y[:train_border])

train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create your datset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  num_workers=16) # create your dataloader


val_tensor_x = torch.Tensor(x[train_border:]) # transform to torch tensor
val_tensor_y = torch.Tensor(y[train_border:])

val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create your datset
val_loader = DataLoader(val_dataset, batch_size=512,  num_workers=16) # create your dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP(512).to(device)   # CLIP embedding dim is 768 for CLIP ViT L 14

optimizer = torch.optim.Adam(model.parameters()) 

# choose the loss you want to optimze for
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

epochs = 10000

model.train()
best_loss =999
save_name = "linear_predictor_L14_MSE.pth"


for epoch in range(epochs):
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())


        optimizer.step()

        if batch_num % 1000 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
            #print(y)

    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    losses = []
    losses2 = []
    
    for batch_num, input_data in enumerate(val_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        lossMAE = criterion2(output, y)
        #loss.backward()
        losses.append(loss.item())
        losses2.append(lossMAE.item())
        #optimizer.step()

        if batch_num % 1000 == 0:
            print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, batch_num, loss.item()))
            print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, batch_num, lossMAE.item()))


    print('Validation - Epoch %d | MSE Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    print('Validation - Epoch %d | MAE Loss %6.2f' % (epoch, sum(losses2)/len(losses2)))
    if sum(losses)/len(losses) < best_loss:
        print("Best MAE Val loss so far. Saving model")
        best_loss = sum(losses)/len(losses)
        print( best_loss ) 

        torch.save(model.state_dict(), save_name )


torch.save(model.state_dict(), save_name)

print( best_loss ) 

print("training done")

test_tensor_x = torch.Tensor(x[train_border:]).to(device)
test_tensor_y = torch.Tensor(y[train_border:]).to(device)
test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

model = MLP(input_size=x.shape[1]).to(device)
model.eval()

output = model(x[:500].to(device))
print(output.size())
print(output)

output_mean = torch.mean(output)
print("Output mean:", output_mean)
