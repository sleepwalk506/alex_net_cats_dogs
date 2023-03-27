import torch
from net import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cuda':
    print('train on cuda')

model = MyAlexNet().to(device)#模型输入到显卡中（如果有的话）
#加载模型
model.load_state_dict(torch.load("D:/deep learning2/hualidefeng_nets/alex_net/save_model/best_model.pth",map_location='cpu'))
# model.torch.load("D:/deep learning2/hualidefeng_nets/alex_net/save_model/best_model.pth",map_location='cpu')
#获取预测结果
classes = [
    "cat",
    "dog"
]
#tensor转化为照片格式
show = ToPILImage()
#进入到验证阶段
model.eval()
for i in range(10):
    x,y = val_dataset[i][0],val_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x,dim=0).float(),requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted,actual = classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",Actual:"{actual}"')

