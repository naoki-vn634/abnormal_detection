import argparse
import os 
from glob import glob
from preprocess import MNISTDataset
import torchvision.transforms as transforms
import numpy as np

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST')
    args = parser.parse_args()

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
    
    print(train_dataset[0])
    print(test_dataset[0])



if __name__ == '__main__':
    main()