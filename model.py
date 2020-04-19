import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim=1,hidden_size=16, out_dim=4):
        super(ConvAutoEncoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(input_dim, hidden_size, kernel_size = 2, padding = 1)
        self.conv2 = nn.Conv2d(hidden_size,hidden_size,kernel_size = 2 ,padding = 1)
        self.conv2 = nn.Conv2d(hidden_size, out_dim, kernel_size = 2, padding = 1)
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