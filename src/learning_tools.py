# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 22:20:07 2022

@author: Schechter
"""
#import libraries
import os
import torch
import numpy as np
import pandas as pd
import slayerSNN as snn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from slayerSNN.learningStats import learningStats
from torch.optim import lr_scheduler as LR


def kfold_split(fileList, seed, export_txt=False):
    """
    Split a file list (txt file) in 4 folds for cross validation (75%, 25%).
    It shuffles the files and then returns 4 separate training and test lists. 
    Optionally export the lists as txt files (default=False).
    
    Returns a generator.
    """
    def gen():  
        #load the files from .txt to an numpy 1D array
        files = np.loadtxt(fileList, dtype='str')  #[max 59 recordings]
        #create KFold object
        kf = KFold(n_splits=4, shuffle=True, random_state=seed)
        #create the folds
        for i, (train_index, test_index) in enumerate(kf.split(files), start=1):
            train_set, test_set = files[train_index], files[test_index]
            if export_txt:
                np.savetxt('../data/trainlist{}.txt'.format(i), train_set, 
                           fmt='%s')
                np.savetxt('../data/testlist{}.txt'.format(i), test_set, 
                           fmt='%s')
            yield train_set, test_set
    return gen()  #returns a generator!


def plot_spike_raster(output, net_params, results_path, fold, label):
    """
    Plot output spike RASTER and HISTOGRAM
    Output batch ('output') shape = (nSample, 19, 1, 1, nTimeBins)
    The plots are from the 1st sample in the output batch (output[0], label[0])
    """
    #read animal classes from file
    class_file = pd.read_csv(net_params['training']['path']['data']
                             + 'SL-Animals-DVS_gestures_definitions.csv')
    
    #unsqueeze if single file on batch
    if len(output) == 1:
        label = label.unsqueeze(0)
    
    #plot spike raster
    plt.figure(figsize=(14,5))
    outAER = np.argwhere(output[0].squeeze().cpu().data.numpy() > 0)
    #outAER shape [total_spikes, 2]; col0=neuronIndex, col1=binNumber
    plt.plot(outAER[:, 1] * net_params['simulation']['Ts'],  #spikeTime
             outAER[:, 0] + 1, '|')                          #neuronNumber
    plt.xticks(range(0, net_params['simulation']['tSample'] + 100, 100))
    plt.yticks(range(1, len(class_file)+1), labels=class_file['class Def'])
    plt.xlabel('Spike Time [ms]', fontsize=14)
    plt.ylabel('Output Neuron [class]', fontsize=14)
    plt.title('Output Spike Raster on test sample (target={})'.format(
        class_file.iloc[label[0].item(), 1]), fontweight='bold', fontsize=14)
    #save the figure
    save_path = results_path + 'output_spike_raster_fold{}.png'.format(fold)
    plt.savefig(save_path)
    
    #plot output spike histogram
    plt.figure(figsize=(8,5))
    total_spikes = outAER[:, 0] + 1
    # spikes_per_class = np.unique(total_spikes, return_counts=True)
    spikes_per_class, _bins, _bars = plt.hist(
        total_spikes, bins=range(1, len(class_file)+2), #range(1, 21)
        align='left', rwidth=0.5, label='actual spike count'
        )
    #plot desired spike count for correct and incorrect classes
    plt.axhline(y=net_params['training']['error']['tgtSpikeCount'][True], 
        color='b', linestyle='--', linewidth=1, label='desired correct')
    plt.axhline(y=net_params['training']['error']['tgtSpikeCount'][False], 
        color='r', linestyle='--', linewidth=1, label='desired incorrect')
    #plot spike count on winning class
    winning_index = np.argmax(spikes_per_class)
    x_loc = winning_index + 0.5      
    y_loc = spikes_per_class[winning_index] + 2
    #or alternatively: x_loc = _bars[winning_index].get_x() - 0.25    
    #                  y_loc = _bars[winning_index].get_height() + 2
    plt.text(x_loc, y_loc, int(spikes_per_class[winning_index]))
    plt.xticks(range(1, len(class_file)+1), labels=class_file['class Def'], 
                rotation=45)  #range(1, 20)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Spikes', fontsize=14)
    plt.title('Output Spike Histogram on test sample (target={})'.format(
        class_file.iloc[label[0].item(), 1]), fontweight='bold', fontsize=14)
    plt.legend(loc='center')
    #save the figure
    save_path = results_path + 'output_spike_histogram_fold{}.png'.format(fold)
    plt.savefig(save_path)
    
    plt.show()    #show both plots


def train_net(net, optimizer, error, train_loader, val_loader, 
              net_params, device, writer, results_path, epochs, 
              fold=None, patience=100, pretrain=False, resume=False):
    
    #define some parameters
    start_time = datetime.now()          #measure total training time
    stats = learningStats()              #learning stats instance.
    no_improvement = 0                   #counter for val. loss improvement
    
    #if using pre-trained weights
    if pretrain:
        print("Loading pre-trained weights...")
        net.load_state_dict(
            torch.load(results_path + 'model_weights_fold{}.pth'.format(fold))
            )
    
    #if continuing training from a checkpoint
    if resume: 
        print("Loading a checkpoint...")
        checkpoint = torch.load(results_path + 'checkpoint{}.tar'.format(fold))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print("Resuming training from epoch", first_epoch)
    else:
        first_epoch = 0
    
    # Main loop
    for epoch in range(first_epoch, epochs):
        epoch_st = datetime.now()        #epoch start time
        
        # Training Loop
        net.train()                      #prep. model for training
        for i, (input, target) in enumerate(train_loader):
            #get a label vector for this batch
            label = np.argmax(target, axis=1).squeeze()
            
            # Send the input and target to the GPU.
            input  = input.to(device)    #input batch
            target = target.to(device)   #target batch
            
            # Feed-Forward
            output = net.forward(input)  #output spike train
            
            # Calculate loss
            loss = error.numSpikes(output, target)
    
            # Gradient BackProp and weight update
            optimizer.zero_grad()        #clear the gradients
            loss.backward()              #backpropagation of error
            optimizer.step()             #weight update
            
            # Training Statistics
            stats.training.numSamples     += len(target)
            stats.training.lossSum        += loss.cpu().data.item()
            stats.training.correctSamples += torch.sum( 
                snn.predict.getClass(output) == label ).data.item()
            
            # Display training stats. (every 10th batch and at last batch)
            if (i%10 == 0 or i+1 == len(train_loader)):   
                stats.print(
                    epoch, i, (datetime.now() - epoch_st).total_seconds()
                    )
                #write statistics to Tensorboard
                writer.add_scalar('Train Loss /batch_idx', 
                                  stats.training.lossSum / stats.training.
                                      numSamples, 
                                  i + len(train_loader) * epoch)
                writer.add_scalar('Train Accuracy /epoch',  
                                  100 * stats.training.correctSamples / stats.
                                      training.numSamples, 
                                  epoch)
            
        # Validation loop.
        net.eval()                       #prep. model for evaluation
        with torch.no_grad():            #do not track gradients on inference
            for i, (input, target) in enumerate(val_loader):
                #get a label vector for this batch
                label = np.argmax(target, axis=1).squeeze()
                #send to device
                input  = input.to(device)    #input batch
                target = target.to(device)   #target batch
                #feed forward
                output = net.forward(input)  #output batch
                #calculate test loss
                val_loss = error.numSpikes(output, target) 
                #statistics
                stats.testing.numSamples     += len(target)
                stats.testing.lossSum        += val_loss.cpu().data.item()
                stats.testing.correctSamples += torch.sum( 
                    snn.predict.getClass(output) == label ).data.item()
                #print control
                if (i%10 == 0 or i+1 == len(val_loader)):
                    stats.print(epoch, i)
        
        #write statistics to Tensorboard
        writer.add_scalar('Val. Loss /epoch', 
                          stats.testing.lossSum / stats.testing.numSamples,
                          epoch)
        writer.add_scalar('Val. Accuracy /epoch',  
                          100 * stats.testing.correctSamples / stats.testing.
                              numSamples, 
                          epoch)
        for (name, param) in net.named_parameters():
            writer.add_histogram(name, param, epoch)
        
        # Update stats.
        stats.update()
        
        #save best weights
        if stats.testing.bestLoss:       #if validation loss reduces
            torch.save(net.state_dict(), 
                       results_path + 'model_weights_fold{}.pth'.format(fold))
            no_improvement = 0           #reset counter
        else:
            no_improvement += 1          #increase counter
            
        #check for early stopping after at least 200 epochs
        if (epoch >= 200 and no_improvement >= patience):
            print('Early stopping after {} epochs!'.format(epoch + 1))
            #save a checkpoint, if later needed to resume training
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, results_path + 'checkpoint{}.tar'.format(fold))
            break
       
        #Reduce LR by schedule at the end of every epoch
        # scheduler.step()

        #end of epoch---------------------------------------
    
    #measure testing time
    end_time = datetime.now()
    print('Total Training Time:', end_time-start_time)
    
    # Plot the results.
    # Learning loss
    plt.figure(1)
    # plt.semilogy(stats.training.lossLog, label='Training')
    # plt.semilogy(stats.testing .lossLog, label='Validation')
    plt.plot(stats.training.lossLog, label='Training')
    plt.plot(stats.testing .lossLog, label='Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(visible=True, axis='y')
    plt.title('Loss using ' + net_params['training']['transf_method'],
              fontweight='bold', fontsize=14)
    #save the figure
    save_path = results_path + 'loss_ws10_batch{}_ADAM_fold{}.png'.format(
        net_params['simulation']['nSample'], fold)
    plt.savefig(save_path)
    
    # Learning accuracy
    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing .accuracyLog, label='Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.grid(visible=True, axis='y')
    plt.title('Accuracy using ' + net_params['training']['transf_method'],
              fontweight='bold', fontsize=14)
    #save the figure
    save_path = results_path + 'accuracy_ws10_batch{}_ADAM_fold{}.png'.format( 
        net_params['simulation']['nSample'], fold)
    plt.savefig(save_path)
    
    plt.show()  #show both plots
    
    return stats


def test_net(net, error, test_loader, net_params, device, writer, results_path, fold=None):
    
    #initialize stats
    stats = learningStats()              #learning stats instance.
    
    #load best weights
    net.load_state_dict(
        torch.load(results_path + 'model_weights_fold{}.pth'.format(fold))
        )

    #Testing loop.
    net.eval()                           #prep. model for testing
    with torch.no_grad():                #do not track gradients on inference
        for i, (input, target) in enumerate(test_loader):
            #get a label vector for this batch
            label = np.argmax(target, axis=1).squeeze()
            #send to device
            input  = input.to(device)    #input batch
            target = target.to(device)   #target batch
            #feed forward
            output = net.forward(input)  #output batch
            #calculate test loss
            loss = error.numSpikes(output, target) 
            #statistics
            stats.testing.numSamples     += len(target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            #print control
            if (i%10 == 0 or i+1 == len(test_loader)):
                stats.print(0, i)  #only "epoch 0"

    #write statistics to Tensorboard
    writer.add_scalar('Test Loss', 
                      stats.testing.lossSum / stats.testing.numSamples)
    writer.add_scalar('Test Accuracy',  
                      100 * stats.testing.correctSamples / stats.testing.
                          numSamples)
    # Update stats.
    stats.update()
    
    #Plot the Results
    #plot output spike raster and histogram on the last sample (testing set)
    plot_spike_raster(output, net_params, results_path, fold, label)

    #export results
    return stats

    #for troubleshooting export also 'output':
    # return output, stats
