import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
train_dataset = torchvision.datasets.MNIST(root='/Users/matsunaganaoki/Desktop/DeepLearning/data',train=True, transform=None)
test_dataset = torchvision.datasets.MNIST(root='/Users/matsunaganaoki/Desktop/DeepLearning/data',train=False, transform=None)

for i in range(len(train_dataset)):
    image,label = train_dataset[i]
    image.save("/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST/train/{}/image_{}.png".format(str(label),i))


for i in range(len(test_dataset)):
    image,label = test_dataset[i]
    image.save("/Users/matsunaganaoki/Desktop/DeepLearning/data/MNIST/test/{}/image_{}.png".format(str(label),i))