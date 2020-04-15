import argparse
import os 
from glob import glob
from preprocess import MNISTDataset
from model import ConvAutoEncoder
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch 
import torchvision


def import_path(dir_path):
    img_list = []
    label = np.empty(0)

    for dir in dir_path:
        num = int(os.path.basename(dir))
        img_path = glob(os.path.join(dir,'*.png'))
        img_path_list = [path for path in img_path]
        img_list.extend(img_path_list)
        for i in range(len(img_path)):
            label = np.append(label,num)

    return img_list,label

def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam, loss_fn = nn.MSELoss(), device = "cpu"):
    

    losses = []         #loss_functionの遷移を記録
    optimizer = optimizer_cls(net.parameters(), lr = 0.001)
    net.to(device)

    for epoch in range(n_epochs):
        running_loss = 0.0  
        net.train()         #ネットワークをtrainingモード

        for i, (image,label) in enumerate(train_loader):
            image.to(device)
            optimizer.zero_grad()
            output = net(image)             #ネットワークで予測
            loss = loss_fn(image, output)   #予測データと元のデータの予測
            loss.backward()
            optimizer.step()              #勾配の更新
            running_loss += loss.item()

        losses.append(running_loss / i)
        print("epoch", epoch, ": ", running_loss / i)

    return losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("#device: ",device)
    ##trainデータは，1,4,7
    
    train_data = [1,4,7]

    train_dir_path = glob(os.path.join(args.input,'train/*'))
    test_dir_path = glob(os.path.join(args.input,'test/*'))

    img_path_train,label_train = import_path(train_dir_path)
    img_path_test,label_test = import_path(test_dir_path)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))])

    train_dataset = MNISTDataset(img_path_train,label_train,transform=transform)
    test_dataset = MNISTDataset(img_path_test,label_test,transform=transform)
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch,shuffle=False)
    model = ConvAutoEncoder(input_dim=3, hidden_size=16, out_dim=4)
    model.to(device)
    

    
    losses = train_net(n_epochs=args.epoch, train_loader=train_dataloader, net=model, optimizer_cls = optim.Adam,
              loss_fn = nn.MSELoss(), device = device)
    # print(train_dataset[0])
    # print(test_dataset[0])



if __name__ == '__main__':
    main()