import torch
from torch import nn
from net import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r"D:/deep learning2/hualidefeng_nets/alex_net/data/train"
ROOT_TEST = r"D:/deep learning2/hualidefeng_nets/alex_net/data/val"

#将图像的像素值归一化到[-1,1]之间
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

train_transform = transforms.Compose([
    transforms.Resize((224,224)),#把图像resize到given size
    transforms.RandomVerticalFlip(),#基于概率的执行图片垂直翻转
    transforms.ToTensor(),#totensor能够把灰度范围从0-255变换到0-1
    normalize#把0-1变换到（-1,1）
])

val_transform = transforms.Compose([#验证集不需要翻转图片
    transforms.Resize((224,224)),#把图像resize到given size
    transforms.ToTensor(),#totensor能够把灰度范围从0-255变换到0-1
    normalize#把0-1变换到（-1,1）
])

train_dataset = ImageFolder(ROOT_TRAIN,transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST,transform=val_transform)

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)#模型输入到显卡中（如果有的话）

#定义一个损失函数
loss_fn = nn.CrossEntropyLoss()
#定义一个优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)#将模型参数传给优化器，学习率
#学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

#定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n = 0.0,0.0,0#定义初始loss值，精确度，指示器
    for batch,(x,y) in enumerate(dataloader):
        image,y = x.to(device),y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)#真实值和预测值做误差
        _,pred = torch.max(output,axis=1)#正确率最多的值取出来
        cur_acc = torch.sum(y==pred)/output.shape[0]#当前准确率

        #反向传播
        optimizer.zero_grad()#先将梯度降为0
        cur_loss.backward()#反向传播
        optimizer.step()#更新梯度
        loss += cur_loss.item()#loss值累加
        current += cur_acc.item()
        n += 1

    train_loss = loss/n
    train_acc = current/n
    print('train_loss:' + str(train_loss))
    print('train_acc:' + str(train_acc))
    return train_loss,train_acc

#定义一个验证函数
def val(dataloader,model,loss_fn):
    model.eval()
    loss,current,n = 0.0,0.0,0#定义初始loss值，精确度，指示器
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)  # 真实值和预测值做误差
            _, pred = torch.max(output, axis=1)  # 正确率最多的值取出来
            cur_acc = torch.sum(y == pred) / output.shape[0]  # 当前准确率
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    val_loss = loss/n
    val_acc = current/n
    print('val_loss:' + str(val_loss))
    print('val_acc:' + str(val_acc))
    return val_loss,val_acc

#开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 1
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()#学习率更改
    print(f"epoch{t+1}\n---------")
    train_loss,train_acc = train(train_dataloader,model,loss_fn,optimizer)
    val_loss,val_acc = val(val_dataloader,model,loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    #保存最好的模型权重
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
            min_acc = val_acc
            print(f"save best model,epoch{t+1}")
            torch.save(model.state_dict(),'save_model/best_model.pth')
    #保存最后一轮的权重文件
    if t == epoch-1:
        torch.save(model.state_dict(),'save_model/last_model.pth')

print('Done')

#定义画图函数
def matplot_loss(train_loss,val_loss):
    plt.plot(train_loss,label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train_loss and val_loss graph')
    plt.show()

#定义画图函数
def matplot_acc(train_acc,val_acc):
    plt.plot(train_acc,label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('train_acc and val_acc graph')
    plt.show()

