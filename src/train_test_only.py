# -*- coding: utf-8 -*-
"""
Implementing the SLAYER training on the SL-Animals-DVS dataset.

Created on Wed Nov 23 21:05:37 2022
Author: Schechter
"""
#import libraries
import os
import torch
import numpy as np
import slayerSNN as snn
#import objects from libraries
from datetime import datetime
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#import objects from other python files
from model import MyNetwork, MLPNetwork
from dataset import AnimalsDvsDataset, AnimalsDvsSliced
from learning_tools import kfold_split, train_net, test_net
from utils import plot_events, animate_events, animate_TD, slice_data


#assert we are on the right working directory
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#define the cuda device to run the code on.
device = torch.device('cuda')

#define network parameters file 
net_params = snn.params("network.yaml")

#initializing variables
seed = net_params['training']['seed']     #fixing the seed
test_losses, test_accuracies = [], []     #initializing test history

#define output directory for the results
results_path = net_params['training']['path']['out']
os.makedirs(results_path, exist_ok=True)

#creating a generator to split the data into 4 folds of train/test files
train_test_generator = kfold_split(net_params['training']['path']['file_list'],
                                   seed)

#----------------------------- MAIN PROGRAM ------------------------------
if __name__ == '__main__':  
    
    #define dataset type
    if net_params['training']['sliced_dataset']:  #if using sliced dataset
        dataset_type = AnimalsDvsSliced
        #check if dataset is already sliced (else, slice the raw data files)
        slice_data(net_params['training']['path']['data'],
                   net_params['training']['path']['file_list'])
    else:
        dataset_type = AnimalsDvsDataset
    
    #print header
    print('Welcome to SLAYER Training!')
    print('Starting 4-fold cross validation (train/test only): Please wait...\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with Tensorboard
        writer = SummaryWriter('./logs/fold{}'.format(fold))
        
        #definining train, val and test Datasets
        training_set = dataset_type(
            dataPath     = net_params['training']['path']['data'],
            fileList     = train_set,
            samplingTime = net_params['simulation']['Ts'],
            sampleLength = net_params['simulation']['tSample'],
            fixedLength  = net_params['training']['fix_length'],
            transfMethod = net_params['training']['transf_method'],
            binMode      = net_params['training']['binning_mode'],
            )
        testing_set = dataset_type(
            dataPath     = net_params['training']['path']['data'],
            fileList     = test_set,
            samplingTime = net_params['simulation']['Ts'],
            sampleLength = net_params['simulation']['tSample'],
            fixedLength  = net_params['training']['fix_length'],
            transfMethod = net_params['training']['transf_method'],
            binMode      = net_params['training']['binning_mode'],
            )
        
        # #------------------- visualize dataset samples -------------------
        # if fold == 1:
        #     #visualize input spikes (plot sample)
        #     sample_plot = plot_events(training_set, sample_index=2)
            
        #     #visualize input spikes (animate sample) - '%matplotlib qt5'
        #     sample_animation = animate_events(training_set, sample_index=2)
        #     sample_animation.save("SL_animals_sample1.gif", writer="pillow")
            
        #     #visualize a colored animation
        #     color_animation = animate_TD(training_set, sample_index=2)
        #     color_animation.save("SL_animals_sample2.gif", writer="pillow")
        # #-----------------------------------------------------------------
        
        #if not using sliced data, cache datasets to disk for faster loading
        if not net_params['training']['sliced_dataset']:
            training_set = DiskCachedDataset(
                training_set, 
                net_params['training']['path']['cache'] + 'train', 
                reset_cache=True)
            testing_set = DiskCachedDataset(
                testing_set, 
                net_params['training']['path']['cache'] + 'test',
                reset_cache=True)
        
        #definining train and test DataLoaders
        """
        If each sample on the dataset has the same length, batch_size can be of
        any size (as long as it fits in the GPU memory). If each sample has 
        different time lengths, batch_size has to be 1! Since Animals-DVS 
        dataset has different sample lengths, the only way to achieve training 
        in batches is to crop all the samples to a fixed length size.
        """
        batchsize = net_params['simulation']['nSample'] if \
                    net_params['training']['fix_length'] else 1

        train_loader = DataLoader(dataset=training_set, batch_size=batchsize,
                                  shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset=testing_set, batch_size=batchsize,
                                  shuffle=False, num_workers=4)
        
        # Create network instance (Slayer SNN) -> send to device
        net = MyNetwork(net_params).to(device) if net_params['training']['CNN'] \
            else MLPNetwork(net_params).to(device)

        # Define optimizer module.
        if net_params['training']['optimizer'] == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), 
                                         lr=net_params['training']['lr'], 
                                         amsgrad=True)
        elif net_params['training']['optimizer'] == 'NADAM':
            optimizer = snn.utils.optim.Nadam(net.parameters(), 
                                              lr=net_params['training']['lr'], 
                                              amsgrad=True)
        elif net_params['training']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=net_params['training']['lr'])
            # scheduler = LR.ExponentialLR(optimizer, 0.95)
        else:
            print("Optimizer option is not valid; using ADAM instead.")
            optimizer = torch.optim.Adam(net.parameters(), 
                                         lr=net_params['training']['lr'], 
                                         amsgrad=True)

        # Create snn loss instance -> send to device
        error = snn.loss(net_params).to(device)

        #train the network
        print('\nTRAINING FOLD {}:'.format(fold))
        print("-----------------------------------------------")
        print('events_transf={}, binning={}, sliced_data={}, batch_size={},'
              .format(net_params['training']['transf_method'], 
                      net_params['training']['binning_mode'],
                      net_params['training']['sliced_dataset'],
                      batchsize))
        print('optimizer={}, initial_lr={}, epochs={}, seed={},'.format(
            net_params['training']['optimizer'],
            net_params['training']['lr'],
            net_params['training']['epochs'],
            net_params['training']['seed']))
        print('simulation length={}ms, simulation step size={}ms'.format(
            net_params['simulation']['tSample'],
            net_params['simulation']['Ts']))
        
        train_stats = train_net(net, optimizer, error, train_loader, test_loader, 
                                net_params, device, writer, results_path, 
                                epochs=net_params['training']['epochs'], fold=fold)
        min_loss = train_stats.testing.minloss
        max_acc  = train_stats.testing.maxAccuracy
        
        #test the network (for plotting purposes only)
        print('\nTESTING FOLD {}:'.format(fold))
        print("-----------------------------------------------")
        test_stats = test_net(net, error, test_loader, net_params, device, writer, 
                              results_path, fold=fold)
        
        #save this fold's losses and accuracies in history
        test_losses.append(min_loss)
        test_accuracies.append(max_acc)

    #end of cross validation---------------------------------------------
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
    
    #print results
    print("\nMin Test Loss on 4 folds:", test_losses)
    print("Average (min) Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(test_losses), np.std(test_losses)))

    print("\nMax Test Accuracy on 4 folds:", test_accuracies)
    print("Average (max) Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        100 * np.mean(test_accuracies), 100 * np.std(test_accuracies)))

# plt.close('all')  #close all open figures
    
    
   
#----------------------------------------------------------------------------
#useful libraries

# import torchsample  #torchsample.callbacks.EarlyStopping
# from torchsummary import summary
# (usage: summary(net, input_size=(C, H, W, T), batch_size=batchsize))
