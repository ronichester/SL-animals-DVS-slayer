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
from dataset import AnimalsDvsDataset, AnimalsDvsSliced
from utils import plot_events, animate_events, animate_TD, slice_data
from learning_tools import kfold_split, train_net, test_net


#assert we are on the right working directory
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#define the cuda device to run the code on.
device = torch.device('cuda')

#define network parameters file 
net_params = snn.params("network.yaml")

#initializing variables
seed = net_params['training']['seed']     #fixing the seed
val_losses, val_accuracies = [], []       #initializing val history
test_losses, test_accuracies = [], []     #initializing test history

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
    print('Welcome to SLAYER Training! Starting 4-fold cross validation...')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with Tensorboard
        writer = SummaryWriter('./summaries/fold{}'.format(fold))
        
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
        
        #**********************************************
        # if fold > 1: # (to make adjusts) DELETE LATER
        # *********************************************
            
        #train the network
        print('\nTRAINING FOLD = {}:'.format(fold))
        print('events_transf={}, batch_size={}, binning={}, sliced_data={}'
              .format(net_params['training']['transf_method'], batchsize,
                      net_params['training']['binning_mode'],
                      net_params['training']['sliced_dataset']))
        print('simulation length={}ms, simulation step size={}ms, seed={}'
              .format(net_params['simulation']['tSample'],
                      net_params['simulation']['Ts'],
                      net_params['training']['seed']))
        train_stats = train_net(train_loader, test_loader, net_params,
                                device, writer, epochs=1000, fold=fold)
        min_loss = train_stats.testing.minloss      #min. val loss = test
        max_acc  = train_stats.testing.maxAccuracy  #max. val accuracy = test
        
        #test the network (for plotting purposes only)
        print('\nStarting Testing of fold {}...'.format(fold))
        test_stats = test_net(test_loader, net_params, device, writer, fold=fold)
        test_loss  = test_stats.testing.minloss       #test loss
        test_acc   = test_stats.testing.maxAccuracy   #test accuracy
            
        #save this fold's losses and accuracies in history
        val_losses.append(min_loss)
        val_accuracies.append(max_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
            
    #end of cross validation---------------------------------------------
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
    
    #print results
    print("\nMin Test Loss on 4 folds:", val_losses)
    print("Min Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(val_losses), np.std(val_losses)))

    print("\nMax Test Accuracy on 4 folds:", val_accuracies)
    print("Max Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        100 * np.mean(val_accuracies), 100 * np.std(val_accuracies)))
    
    #redundant here, test_losses should be equal val_losses; just sanity check
    print("\nTest Loss on 4 folds:", test_losses)
    print("Average Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(test_losses), np.std(test_losses)))

    print("\nTest Accuracy on MIN LOSS (4 folds):", test_accuracies)
    print("Average Test Accuracy on MIN LOSS:     {:.2f}% +- {:.2f}%".format(
        100 * np.mean(test_accuracies), 100 * np.std(test_accuracies)))

# plt.close('all')  #close all open figures
    
    
   
#----------------------------------------------------------------------------
#useful libraries

# import torchsample  #torchsample.callbacks.EarlyStopping
# from torchsummary import summary
# (usage: summary(net, input_size=(C, H, W, T), batch_size=batchsize))