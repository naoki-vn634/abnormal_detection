
import os 
import cv2
import torch 
import argparse
import torchvision

import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn as nn

from glob import glob
from preprocess import MNISTDataset
from model import ConvAutoEncoder
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

def list_compile(label,loss):
    dict = {}
    
    
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


def eval_net(out,test_loader,net,device):
    loss_fn = nn.MSELoss()
    loss_dict = {}

    
    for i,(image, label) in enumerate(test_loader):
        image.to(device)
        output = net(image)
        for j in range(len(label)):
            loss = loss_fn(image[j],output[j])
            pred = 0.5*(output[j]+1)*255 
            pred_img = pred.detach().numpy()
            pred_img = pred_img.transpose(1,2,0)
            pred_img = pred_img.astype(np.uint8)
            num = label[j]
            num_str = str(int(num))
            
            loss_dict.setdefault(num_str,[]).append(float(loss.tolist()))
            
            # pred_img.save(os.path.join(out,'{}/result_{}.png'.format(str(int(num)),i)))

            cv2.imwrite(os.path.join(out,'{}/result_{}.png'.format(str(int(num)),i)),pred_img)
        
        # if i > 200:
        #     break
    
    loss_0 = np.sort(np.array(loss_dict["0"]))
    loss_1 = np.sort(np.array(loss_dict["1"]))
    loss_2 = np.sort(np.array(loss_dict["2"]))
    loss_3 = np.sort(np.array(loss_dict["3"]))
    loss_4 = np.sort(np.array(loss_dict["4"]))
    loss_5 = np.sort(np.array(loss_dict["5"]))
    loss_6 = np.sort(np.array(loss_dict["6"]))
    loss_7 = np.sort(np.array(loss_dict["7"]))
    loss_8 = np.sort(np.array(loss_dict["8"]))
    loss_9 = np.sort(np.array(loss_dict["9"]))
    
    plt.figure()
    sns.distplot(loss_1, hist=False,label="1")
    sns.distplot(loss_4, hist=False,label="4")
    sns.distplot(loss_7, hist=False,label="7")
    plt.savefig("result_train.png")
    
    plt.figure()
    sns.distplot(loss_0, hist=False,label="0")
    sns.distplot(loss_2, hist=False,label="2")
    sns.distplot(loss_3, hist=False,label="3")
    sns.distplot(loss_5, hist=False,label="5")
    sns.distplot(loss_6, hist=False,label="6")
    sns.distplot(loss_8, hist=False,label="8")
    sns.distplot(loss_9, hist=False,label="9")
    plt.savefig("result_test.png")
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST')
    parser.add_argument('--output',type=str, default='/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST/result')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument('--weight',type=str, default='/Users/matsunaganaoki/Desktop/DeepLearning/data/weight/epoch_49_losses_0.8935579522457512_abnormal_network.pth')
    args = parser.parse_args()
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("#device: ",device)
    
    test_dir_path = glob(os.path.join(args.input,'test/*'))
    img_path_test,label_test = import_path(test_dir_path)
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
    
    test_dataset = MNISTDataset(img_path_test,label_test,transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch,shuffle=True)
    
    model = ConvAutoEncoder(input_dim=1,hidden_size=16,out_dim=4)
    model.load_state_dict(torch.load(args.weight,map_location='cpu'))
    
    model.to(device)
    
    eval_net(out=args.output,test_loader=test_dataloader,net=model,device=device)
    
    
if __name__ == '__main__':
    main()