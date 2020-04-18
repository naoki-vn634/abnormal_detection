import torch
import torchvision.transforms as transforms
import cv2

from PIL import Image
import numpy as np


class MNISTDataset(object):
    def __init__(self,img_path,label,transform):
        self.file_path = img_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self,index):
        # img = cv2.imread(self.file_path[index])
        img = Image.open(self.file_path[index])
        img = np.array(img,np.uint8)
        img = np.expand_dims(img,2)
        # print(img.shape)
        img_transformed = self.transform(img)
        return img_transformed,self.label[index]