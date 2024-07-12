"""
Created on Mon Nov 29 16:06:50 2022

The Spiking Neural Network (SNN) - model.py
    MLP: Two Fully Connected Layers (Dense)
    CNN: 3 pooling, 2 convolutions, 2 dense layers

@author: Schechter
"""
#import libraries
import torch
import slayerSNN as snn

# Network definition ('SLAYER' type of SNN)
class MLPNetwork(torch.nn.Module):
    def __init__(self, netParams):
        super(MLPNetwork, self).__init__()
        # Initialize slayer
        # snn.layer is a shortcut for snn.slayer.spikeLayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Define network functions
        # The line below should be used if input spikes were not reshaped
        # self.fc1   = slayer.dense((128, 128, 2), 512)
        self.fc1   = slayer.dense((128*128*2), 512)  #Dense(512)
        self.fc2   = slayer.dense(512, 19)           #Dense(19)

    def forward(self, spikeInput):  #forward the input spike train
        """
        spikeInput: Torch Tensor of type float32 shaped [batch-CHWT]
        
        To increase speed, it's recommended to ignore the spatial dimension and 
        place neurons in the channel (C) dimension. [B, C*H*W, 1, 1, T]
        """
        #reshaping spikeInput to increase speed
        B, C, H, W, T = spikeInput.shape   
        spikeInput = spikeInput.reshape((B, C*H*W, 1, 1, T))
        #spike train after crossing layer 1 (fc1)
        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        #spike train after crossing layer 2 (fc2)
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))     
        
        return spikeLayer2  #the output spike train

# Network definition ('SLAYER' type of SNN)
class MyNetwork(torch.nn.Module):
    def __init__(self, netParams):
        super(MyNetwork, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        #self.params = netParams
        # define network layers
        self.pool1 = slayer.pool(2)
        self.conv1 = slayer.conv(2, 8, 5, padding=2, weightScale=10)
        self.pool2 = slayer.pool(2)
        self.conv2 = slayer.conv(8, 16, 5, padding=2, weightScale=10)
        self.pool3 = slayer.pool(2)
        # self.fc1   = slayer.dense((16, 16, 16), 100) #if not reshaped
        self.fc1   = slayer.dense((16*16*16), 100)     #if reshaped
        self.fc2   = slayer.dense(100, 19)
        # self.drop  = slayer.dropout(0.3)             #### 30% dropout ####
        
    def forward(self, spikeInput):
        """
        spikeInput: Torch Tensor of type float32 shaped [batch-CHWT]
                    It's the input spike train
        spike:      Hidden layer spike train
        spikeOut:   Torch Tensor of type float32 shaped [batch-CHWT]
                    It's the output spike train
        
        To increase speed, for the fully connected layers it's recommended to 
        ignore the spatial dimension and place neurons in the channel (C) 
        dimension. [B, C*H*W, 1, 1, T]
        """
        #spikeInput shape 2x128x128
        spike    = self.slayer.spike(self.slayer.psp(self.pool1(spikeInput)))          #POOL: output 2x64x64
    
        spike    = self.slayer.spike(self.slayer.psp(self.conv1(spike)))               #CONV: output 8x64x64
        spike    = self.slayer.spike(self.slayer.psp(self.pool2(spike)))               #POOL: output 8x32x32 
        
        spike    = self.slayer.spike(self.slayer.psp(self.conv2(spike)))               #CONV: output 16x32x32
        spike    = self.slayer.spike(self.slayer.psp(self.pool3(spike)))               #POOL: output 16x16x16
        #reshaping the spikes to increase speed, before FC (Dense) layers                      
        spike    = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))          #RESHAPE: output 4096x1x1
        # spike    = self.drop(spike)                                                    #DROPOUT: used 0.1 in Loihi/IBM-gestures
        
        spike    = self.slayer.spike(self.slayer.psp(self.fc1  (spike)))               #DENSE: output 100x1x1
        spikeOut = self.slayer.spike(self.slayer.psp(self.fc2  (spike)))               #DENSE: output 19x1x1
        
        return spikeOut