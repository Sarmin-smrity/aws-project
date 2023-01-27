#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
os.environ['TORCH_HOME'] = os.getcwd() + "/torch_home/" 
from torchvision.datasets import ImageFolder
#import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, roc_auc_score, roc_curve
import pandas as pd
#import os
import seaborn as sns
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt
transforms = transforms.Compose([
    transforms.Resize(256),    
    transforms.CenterCrop(224),   
    transforms.ToTensor(),       
])

def main(num):
  # Data path
  path = r'/mnt/efs/data/FSL'

  # Set the device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Set the batch size and number of epochs
  batch_size = 24 #12
  num_epochs = 10 #100
  data_train = datasets.ImageFolder(path, transform=transforms)
  data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
  print(f"Data Loaded from : {path}")

  # Define the RESNET model
  #model=torchvision.models.resnet18(pretrained=True)
  model=torchvision.models.AlexNet()
  #model=torchvision.models.vgg11()
  for para in model.parameters():
      para.require_grad=False
  model.fc=torch.nn.Linear(512,2)

  # Move the model to the device
  model.to(device)
  model = DataParallel(model)
  model.train()

  # Define the loss function and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  # Set the training loss list
  train_losses = []

  # Set the training accuracy list
  train_accuracies = []

  print("Traning the model...")
  # Train the model
  for epoch in range(num_epochs):
      if(epoch%10==0):
          print(f'{epoch}',end=' ')
      running_train_loss = 0.0
      running_train_accuracy = 0.0

      for i, (images, labels) in enumerate(data_loader):
          # Move the images and labels to the device
          images = images.to(device)
          labels = labels.to(device)
          predictions = model(images)
          loss = loss_fn(predictions, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          _, predicted_classes = torch.max(predictions, dim=1)
          accuracy = (predicted_classes == labels).float().mean().item()
          running_train_loss += loss.item()
          running_train_accuracy += accuracy
          
      # Calculate the average training loss and accuracy
      train_loss = running_train_loss / len(data_loader)
      train_losses.append(train_loss)
      train_accuracy = running_train_accuracy / len(data_loader)
      train_accuracies.append(train_accuracy)
      if(epoch%10==0):
          print(f'train_loss:{train_loss}',end='  ')
          print(f'train_accuracy:{train_accuracy}')

  #torch.save(model.state_dict(), f'/mnt/efs/modeling/resnet18/resnet_model_{num}.pt')
  torch.save(model.state_dict(), f'/mnt/efs/modeling/alexnet/alexnet_model_{num}.pt')
  #torch.save(model.state_dict(), f'/mnt/efs/modeling/vgg/vgg_model_{num}.pt')
  
  
if __name__ == "__main__":
  if len(sys.argv) == 2:
    number = int(sys.argv[1])
    main(number)
  else:
     print("Invalid number of arguments. Usage: python alexnet.py number")