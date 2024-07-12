# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:06:50 2022

SL-Animals-DVS dataset

@author: Schechter
"""
import numpy as np
import pandas as pd
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import Dataset
from utils import events_to_TD, list_sliced_files

#sliced dataset definition
class AnimalsDvsSliced(Dataset):
    """
    The sliced Animals DVS dataset. Much faster loading and processing!
    """
    
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 transfMethod, randomCrop, binMode):
        
        self.path = dataPath                               #string
        self.slicedDataPath = dataPath + 'sliced_recordings/'   #string
        self.files = list_sliced_files(np.loadtxt(fileList, dtype='str')) #list [1121 files]
        self.samplingTime = samplingTime                   #5 [ms]
        self.sampleLength = sampleLength                   #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime)  #300 bins 
        self.transfMethod = transfMethod                   #string
        self.randomCrop = randomCrop                       #boolean
        self.binMode = binMode                             #string
        #read class file
        self.classes = pd.read_csv(                        #DataFrame
            self.path + 'SL-Animals-DVS_gestures_definitions.csv')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        #load the sample file (NPY format), class name and index
        events, class_name, class_index, ss = self.get_sample(index)
        
        #prepare a target Tensor (class)
        desired_class = torch.zeros((19, 1, 1, 1))  #initialize class tensor
        desired_class[class_index,...] = 1          #set class tensor
        
        #process the events
        if self.transfMethod == 'SlayerTD':
            """
            Use this method with native Slayer TD.
            """
            #transform array of events into Slayer events object (TD)
            chosen_TD = events_to_TD(events)
            #transform Slayer events into Spike Tensor for SNN processing
            input_spikes = chosen_TD.toSpikeTensor(  #shape CHWT
                torch.zeros((2, ss[0], ss[1], self.nTimeBins)),
                samplingTime=self.samplingTime,      #in ms
                # binningMode=self.binMode,            #default = 'OR'
                )
        
        elif self.transfMethod == 'TonicFrames':
            """
            Use this method with Tonic frames.
            """
            #process the events
            frame_transform = transforms.Compose([
                transforms.Downsample(time_factor=0.001),  #us to ms
                transforms.TimeAlignment(),                #1st event at t=0
                transforms.ToFrame(                        #events -> frames
                    sensor_size = (ss[0], ss[1], 2),
                    time_window=self.samplingTime,  #in ms
                    ),
                ])
            
            #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
            frames = frame_transform(events)
                        
            """
            The 'frames' above has variable length for each sample.
            However, SLAYER needs a fixed length in order to train in batches.
            This is achieved (by default) by cropping the samples into fixed 
            crops of 1500 ms, starting from t=0. This implementation offers an
            option of using random sample crops.
            
            Information to be taken into consideration for the SL-Animals:
                - shortest sample: 880 ms. 
                - largest sample: 9466 ms
                - mean sample: 4360 +- 1189 ms stdev.
            """
            if self.randomCrop:  #choose a random crop
                actual_bins = frames.shape[0]      #actual sample length
                bin_diff = actual_bins - self.nTimeBins  #difference
                min_timebin = 0 if bin_diff <= 0 else np.random.randint(0, bin_diff)
                max_timebin = min_timebin + self.nTimeBins
                frames = frames[min_timebin:max_timebin, ...]
            else:                #crop from the beginning
                frames = frames[:self.nTimeBins, ...]
            
            #assure sample has nTimeBins (or pad with zeros)
            if frames.shape[0] < self.nTimeBins:
                padding = np.zeros((self.nTimeBins - frames.shape[0], 
                                    2, ss[0], ss[1]))
                frames = np.concatenate([frames, padding], axis=0)
                
            #input spikes need to be float Tensors shaped CHWT for SLAYER
            frames = frames.transpose(1,3,2,0)   #TCWH -> CHWT
            input_spikes = torch.Tensor(frames)  #torch.float32

            #choice of binning mode
            """
            By default, Tonic sets the number of spikes at each pixel for every
            time bin as an integer number. SLAYER uses values up to '1.0' in
            the Tensors natively, but we can try here 3 types of binning mode:
            -type "OR": if there is either 1 OR more spikes at a specific 
                [x,y] pixel at the same time bin, we set its value fixed at 
                "1.0 / dt" (Slayer's default mode);
            -type "SUM": set the number of spikes at each pixel for every
                time bin as an integer number (Tonic's default mode).
            -type "SUM_NORM": if there is 1 OR more spikes at a specific [x,y] 
                pixel at the same time bin, we set a value proportional to 
                the number of spikes, and so limited to the range [0.0, 1.0];
            """
            if self.binMode == 'OR' :
                #set all pixels with spikes to the value '1.0'
                input_spikes = torch.where(
                    (input_spikes > 0),                 #if spike:
                    1.0,                                #set pixel value to 1
                    input_spikes)                       #else keep value 0
            elif self.binMode == 'SUM' :
                #pixels display the number of spikes (integer) on each time bin
                pass  #do nothing, TonicFrames works natively in 'SUM' mode
            elif self.binMode == 'SUM_NORM' :
                #set all pixels with spikes to a normalized SUM value
                input_spikes = torch.where(
                    (input_spikes > 0),                 #if spike:
                    input_spikes / input_spikes.max(),  #set pixel to range [0, 1.0]
                    input_spikes)                       #else keep value 0
            else:
                print("Invalid binning mode; results are compromised!")
                print("(binning_mode should be only 'OR', 'SUM' or 'SUM_NORM')")
        
        else:
            print("Invalid transform method from events to Tensor of spikes!")
            print("('transf_method' should be 'SlayerTD' or 'TonicFrames')")
        
        return input_spikes, desired_class
        
    
    def get_sample(self, index):
        #return the sample events, class name and class index of a sample
        assert index >= 0 and index <= 1120
   
        #the sample file name
        input_name  = self.files[index]
        
        #load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)
        
        #find sample class
        class_index = index % 19                           #[0-18]
        class_name =  self.classes.iloc[class_index, 1]
        
        sensor_shape = (128, 128)
        
        return events, class_name, class_index, sensor_shape
    

# Dataset definition
class AnimalsDvsDataset(Dataset):
    """
    SL-Animals-DVS: event-driven sign language animals dataset
    
    Original paper by Ajay Vasudevan, Pablo Negri, Camila Di Ielsi, Bernabe 
    Linares‑Barranco, Teresa Serrano‑Gotarredona.
    """
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 transfMethod, randomCrop, binMode):
        
        self.path = dataPath                            #string
        self.files = np.loadtxt(fileList, dtype='str')  #1D array [max 59]
        self.samplingTime = samplingTime                #5 [ms]
        self.sampleLength = sampleLength                #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime) #300 bins 
        self.transfMethod = transfMethod                #string
        self.randomCrop = randomCrop                    #boolean
        self.binMode = binMode                          #binning mode
        #read class file
        self.classes = pd.read_csv(
            self.path + 'SL-Animals-DVS_gestures_definitions.csv')
        
    def __len__(self):
        return self.files.shape[0] * len(self.classes)  #'n' samples
    
    def __getitem__(self, index):
        #the relevant events and class name/index for the chosen sample [us]
        events, class_name, class_index, sensor_shape = self.get_sample(index)
        
        #process the events
        if self.transfMethod == 'SlayerTD':
            """
            Use this method with native Slayer TD.
            """
            #transform array of events into Slayer events object (TD)
            chosen_TD = events_to_TD(events)
            #transform Slayer events into Spike Tensor for SNN processing
            input_spikes = chosen_TD.toSpikeTensor(  #shape CHWT
                torch.zeros((2, sensor_shape[0], sensor_shape[1], self.nTimeBins)),
                samplingTime=self.samplingTime,      #in ms
                # binningMode=self.binMode,            #default = 'OR'
                )
        elif self.transfMethod == 'TonicFrames':
            """
            Use this method with Tonic frames.
            """
            #process the events
            frame_transform = transforms.Compose([
                transforms.Downsample(time_factor=0.001),  #us to ms
                transforms.TimeAlignment(),                #1st event at t=0
                transforms.ToFrame(                        #events -> frames
                    sensor_size = (sensor_shape[0], sensor_shape[1], 2),
                    time_window=self.samplingTime,  #in ms
                    ),
                ])
            
            #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
            frames = frame_transform(events)
                        
            """
            The 'frames' above has variable length for each sample.
            However, SLAYER needs a fixed length in order to train in batches.
            This is achieved (by default) by cropping the samples into fixed 
            crops of 1500 ms, starting from t=0. This implementation offers an
            option of using random sample crops.
            
            Information to be taken into consideration for the SL-Animals:
                - shortest sample: 880 ms. 
                - largest sample: 9466 ms
                - mean sample: 4360 +- 1189 ms stdev.
            """
            if self.randomCrop:  #choose a random crop
                actual_bins = frames.shape[0]      #actual sample length
                bin_diff = actual_bins - self.nTimeBins  #difference
                min_timebin = 0 if bin_diff <= 0 else np.random.randint(0, bin_diff)
                max_timebin = min_timebin + self.nTimeBins
                frames = frames[min_timebin:max_timebin, ...]
            else:                #crop from the beginning
                frames = frames[:self.nTimeBins, ...]
            
            #assure sample has nTimeBins (or pad with zeros)
            if frames.shape[0] < self.nTimeBins:
                padding = np.zeros((self.nTimeBins - frames.shape[0], 
                                    2, sensor_shape[0], sensor_shape[1]))
                frames = np.concatenate([frames, padding], axis=0)
                
            #input spikes need to be float Tensors shaped CHWT for SLAYER
            frames = frames.transpose(1,3,2,0)   #TCWH -> CHWT
            input_spikes = torch.Tensor(frames)  #torch.float32
            
            #choice of binning mode
            """
            By default, Tonic sets the number of spikes at each pixel for every
            time bin as an integer number. SLAYER uses values up to '1.0' in
            the Tensors natively, but we can try here 3 types of binning mode:
            -type "OR": if there is either 1 OR more spikes at a specific 
                [x,y] pixel at the same time bin, we set its value fixed at 
                "1.0 / dt" (Slayer's default mode);
            -type "SUM": set the number of spikes at each pixel for every
                time bin as an integer number (Tonic's default mode).
            -type "SUM_NORM": if there is 1 OR more spikes at a specific [x,y] 
                pixel at the same time bin, we set a value proportional to 
                the number of spikes, and so limited to the range [0.0, 1.0];
            """
            if self.binMode == 'OR' :
                #set all pixels with spikes to the value '1.0'
                input_spikes = torch.where(
                    (input_spikes > 0),                 #if spike:
                    1.0,                                #set pixel value to 1
                    input_spikes)                       #else keep value 0
            elif self.binMode == 'SUM' :
                #pixels display the number of spikes (integer) on each time bin
                pass  #do nothing, TonicFrames works natively in 'SUM' mode
            elif self.binMode == 'SUM_NORM' :
                #set all pixels with spikes to a normalized SUM value
                input_spikes = torch.where(
                    (input_spikes > 0),                 #if spike:
                    input_spikes / input_spikes.max(),  #set pixel to range [0, 1.0]
                    input_spikes)                       #else keep value 0
            else:
                print("Invalid binning mode; results are compromised!")
                print("(binning_mode should be only 'OR', 'SUM' or 'SUM_NORM')")
        
        else:
            print("Invalid transform method from events to Tensor of spikes!")
            print("('transf_method' should be 'SlayerTD' or 'TonicFrames')")
         
        #prepare a target Tensor (class)
        desired_class = torch.zeros((19, 1, 1, 1))  #initialize class tensor
        desired_class[class_index,...] = 1          #set class tensor
        
        return input_spikes, desired_class
    
    def get_sample(self, index):
        """
        Get one dataset sample by it's index. The dataset has 1121 samples
        identified by 59 groups (record files) with 19 samples in each group, 
        corresponding to the 19 classes in order.
        
        Parameters:
            index: int, must be between [0, 1120]
        
        Returns:
            chosen_events: 1D array, the sequential events in xytp format [ms]
            class_name: str, the sample class name
            class_index: int, the sample class index [0, 18]
            sensor_shape: tuple of int, (128, 128)
        """
        assert index >= 0 and index <= 1120
        
        #find sample class
        class_index = index % 19
        class_name =  self.classes.iloc[class_index, 1]
        
        #the DVS file name (without suffix)
        input_name  = self.files[index//19]
        
        #read the DVS file
        """
        The output of this function:
            sensor_shape: tuple, the DVS resolution (128, 128)
            events: 1D-array, the sequential events on the file
                    1 microsecond resolution
                    each event is 4D and has the shape 'xytp'
        """
        sensor_shape, events = tonic.io.read_dvs_128(self.path + 'recordings/'
                                                     + input_name + '.aedat')
        #read the tag file
        tagfile = pd.read_csv(self.path + 'tags/' + input_name + '.csv')  #df
        
        #define event boundaries for each class
        events_start = list(tagfile["startTime_ev"])
        events_end = list(tagfile["endTime_ev"])
        
        #create a list of arrays, separating the recording in 19 slices
        sliced_events = tonic.slicers.slice_events_at_indices(events, 
                                                             events_start, 
                                                             events_end)
        #the relevant events for the chosen sample
        chosen_events = sliced_events[class_index]
        
        return chosen_events, class_name, class_index, sensor_shape