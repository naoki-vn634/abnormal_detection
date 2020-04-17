import torch
import torchvision.transforms as transforms
import cv2



class MNISTDataset(object):
    def __init__(self,img_path,label,transform):
        self.file_path = img_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self,index):
        img = cv2.imread(self.file_path[index])
        if img.shape[2]==3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
        img_transformed = self.transform(img)
        return img_transformed,self.label[index]