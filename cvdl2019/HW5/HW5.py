# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

data_path = "CIFAR-10\\cifar-10-python"
batch_size = 64
learning_rate = 0.01
optimizer = "SGD"

trans = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=trans)
test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=trans)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

pred_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False)

image_list = []
label_list = []
for image, label in pred_loader:
    image_list.append(image)
    label_list.append(label.item())
    

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#pred_model = torch.load('net_params3.pkl')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1)
        self.fc1 = nn.Linear(in_features=5*5*16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        x = self.conv1(x)         # [3*32*32] -> [6*28*28]
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [6*28*28] -> [6*14*14]
        x = self.conv2(x)         # [6*14*14] -> [16*10*10]
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [16*10*10] -> [16*5*5]
        x = x.view(x.size(0), -1) # [16*5*5] -> [400]
        x = self.fc1(x)           # [400] -> [120]
        x = F.relu(x)
        x = self.fc2(x)           # [120] -> [84]
        x = F.relu(x)
        x = self.fc3(x)           # [84] -> [10]
        return x

def show_images():
    print("5.1 show train images")
    cnt = 0
    plt.figure()
    for image, label in train_loader:
        if cnt>=10:
            break
        ax = plt.subplot2grid((2, 5), (int(cnt/5), int(cnt%5)), colspan=1, rowspan=1)
        img = image[0]
        img = img.numpy()
        img = np.transpose(img, (1,2,0))
    
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(classes[label[0]])
    
        cnt += 1
    plt.show()

def show_hyperparameters():
    print("==5.2==============")
    print("Hyperparameters")
    print("batch size : " + str(batch_size))
    print("learning rate : " + str(learning_rate))
    print("optimizer : " + optimizer)
    print("===================")
    
def training(self, epochs=1):
    ## training
    print("5.3 train 1 epoch")
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.NLLLoss()
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_train_loss = []
    epoch_test_loss = []
    for epoch in range(epochs):
        # trainning
        train_loss_list = []
        train_corr_count = 0
        train_total_count = 0
        train_loss = 0
        for batch, (x, target) in enumerate(train_loader):
            
            x, target = Variable(x), Variable(target)
            out = model(x)
            
            loss = loss_function(out, target)
            train_loss += loss.data.item()
            train_loss_list.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred_label = torch.max(out.data, 1)
            train_total_count += x.data.size()[0]
            train_corr_count += (pred_label == target.data).sum().item()
            
            if (batch+1) % 1 == 0 or (batch+1) == len(train_loader):
                print ('\r ==== epoch: {}, batch index: {}, acc: {:.3f}, loss: {:.5f}'.format(epoch, batch+1, train_corr_count*1.0/train_total_count, loss.data), end="")
        print()
        epoch_train_acc.append(train_corr_count/train_total_count)
        epoch_train_loss.append(train_loss/len(train_loader))
        # testing
        test_loss_list = []
        test_corr_count = 0
        test_total_count = 0
        test_loss = 0
        for batch, (x, target) in enumerate(test_loader):
            x, target = Variable(x), Variable(target)
            out = model(x)
            
            loss = loss_function(out, target)
            test_loss += loss.data.item()
            test_loss_list.append(loss.data)
            
            _, pred_label = torch.max(out.data, 1)
            test_total_count += x.data.size()[0]
            test_corr_count += (pred_label == target.data).sum().item()
            if(batch+1) % 1 == 0 or (batch+1) == len(test_loader):
                print ('\r ==== epoch: {}, batch index: {}, acc: {:.3f}, test loss: {:.5f}'.format(epoch, batch+1, test_corr_count * 1.0 / test_total_count, loss.data), end='')
        print()
        epoch_test_acc.append(test_corr_count/test_total_count)
        epoch_test_loss.append(test_loss/len(test_loader))
        
        print(epoch_train_acc[-1], epoch_train_loss[-1], epoch_test_acc[-1], epoch_test_loss[-1])
        
        x_list = []
        x_list = np.arange(len(train_loss_list))
        # for item in range(len(train_loss_list)):
        #     x_list.append(int(item))
        plt.plot(x_list, train_loss_list)
        plt.show()

        
def show_train():
    acc = plt.imread("acc.png")
    loss = plt.imread("loss.png")
    
    fig, axes = plt.subplots(1,2,figsize=(13,6)) # set the size that you'd like (width, height)
    ax0, ax1 = axes.ravel()
    ax0.imshow(acc)
    ax0.axis('off')
    ax1.imshow(loss)
    ax1.axis('off')
    
    plt.show()

pred_model = LeNet()
pred_model.load_state_dict(torch.load('net_params10_0.667.pkl'))

def predict(index):
#     pred_model = torch.load('net3.pkl')
    if(index):
        index = int(index)
    else : return
    prediction = pred_model(image_list[index])
    _, pred_label = torch.max(prediction.data, 1)
    print(classes[label_list[index]], classes[pred_label[0].item()])
    
    softmax = nn.Softmax()
    prediction = softmax(prediction)
    prediction = [prediction[0].data[i].item() for i in range(10)]
    print((prediction))
    
    img = image_list[index]
    img = img.numpy()
    img = np.squeeze(img)
    img = np.transpose(img, (1,2,0))

#     plt.subplot(1,2,1)
#     plt.imshow(img)
# #     plt.show()

#     plt.bar(classes,prediction,)
#     plt.show()
    
    fig, axes = plt.subplots(1,2,figsize=(11,5)) # set the size that you'd like (width, height)
    ax0, ax1 = axes.ravel()
    ax0.imshow(img)
    ax1.bar(classes,prediction)
    
    plt.show()

                