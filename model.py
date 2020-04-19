import torch
import torch.nn as nn


class FCAutoencoder(nn.Module):
    
    def __init__(self):
        super(FCAutoencoder, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,16)
        self.fc4 = nn.Linear(16,2) 
        
        self.fc5 = nn.Linear(2,16)
        self.fc6 = nn.Linear(16,64)
        self.fc7 = nn.Linear(64,128)
        self.fc8 = nn.Linear(128,28*28)   
        self.relu = nn.ReLU()
        self.dr = nn.Dropout()
        self.tan = nn.Tanh()
    
    def encoder(self,x):
        x = x.view(len(x),-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        return x
    
    def decoder(self,y):
        y = self.relu(self.fc5(y))
        y = self.relu(self.fc6(y))
        y = self.relu(self.fc7(y))
        y = self.dr(self.relu(self.fc8(y)))
        y = self.tan(y)
        y = y.view(len(y),-1,28,28)
        return y

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim=1,hidden_size=16, out_dim=4):
        super(ConvAutoEncoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(input_dim, hidden_size, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(hidden_size, out_dim, kernel_size = 3, padding = 1)
        #Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(out_dim, hidden_size, kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(hidden_size, input_dim, kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))      
        x = self.pool(x)                  
        x = self.relu(self.conv2(x))      
        x = self.pool(x)                  
        x = self.relu(self.t_conv1(x))    
        x = self.sigmoid(self.t_conv2(x)) 
        return x