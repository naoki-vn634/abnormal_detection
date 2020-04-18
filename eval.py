
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
    for i,(image, label) in enumerate(test_loader):
        image.to(device)
        output = net(image)
        for j in range(len(label)):
            pred = 0.5*(output[j]+1)*255 
            pred_img = pred.detach().numpy()
            pred_img = pred_img.transpose(1,2,0)
            # pred_img = cv2.cvtColor(pred_img,cv2.COLOR_BGR2GLAY)
            pred_img = pred_img.astype(np.uint8)
            
            num = label[j]
#             cv2.imwrite(os.path.join(out,'{}/raw_{}.png'.format(str(int(num)),i)),raw_img)
            cv2.imwrite(os.path.join(out,'{}/result_{}.png'.format(str(int(num)),i)),pred_img)
        
        if i > 200:
            break
        
    
    

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
    
    model = ConvAutoEncoder(input_dim=3,hidden_size=16,out_dim=4)
    model.load_state_dict(torch.load(args.weight,map_location='cpu'))
    
    model.to(device)
    
    eval_net(out=args.output,test_loader=test_dataloader,net=model,device=device)
    
    
if __name__ == '__main__':
    main()