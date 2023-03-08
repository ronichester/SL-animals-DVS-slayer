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
    Make sure to run "slice_data.py" for the 1st time before using this
    dataset to slice and save the files in the correct path.
    """
    
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 fixedLength, transfMethod, binMode):
        
        self.path = dataPath                               #string
        self.slicedDataPath = dataPath + 'sliced_recordings/'   #string
        self.files = list_sliced_files(fileList)           #list [1121 files]
        self.samplingTime = samplingTime                   #5 [ms]
        self.sampleLength = sampleLength                   #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime)  #300 bins 
        self.fixedLength = fixedLength                     #boolean
        self.transfMethod = transfMethod                   #string
        self.binMode = binMode                             #string
        #read class file
        self.classes = pd.read_csv(                        #DataFrame
            self.path + 'SL-Animals-DVS_gestures_definitions.csv')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        assert index >= 0 and index <= 1120
   
        #the sample file name
        input_name  = self.files[index]
        
        #load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)
        
        #find sample class
        class_index = index % 19                           #[0-18]
        # class_name =  self.classes.iloc[class_index, 1]
        
        #prepare a target Tensor (class)
        desired_class = torch.zeros((19, 1, 1, 1))  #initialize class tensor
        desired_class[class_index,...] = 1          #set class tensor
        
        #process the events
        if self.transfMethod == 'SlayerTD':
            """
            Use this method with native Slayer TD (fixed time_bins only)
            """
            #transform array of events into Slayer events object (TD)
            chosen_TD = events_to_TD(events)
            #transform Slayer events into Spike Tensor for SNN processing
            input_spikes = chosen_TD.toSpikeTensor(  #shape CHWT
                torch.zeros((2, 128, 128, self.nTimeBins)),
                samplingTime=self.samplingTime,      #in ms
                # binningMode=self.binMode,            #default = 'OR'
                )
        
        elif self.transfMethod == 'TonicFrames':
            """
            Use this method with Tonic frames (fixed OR variable time_bins).
            """
            #if using fixed sampleLength/time_bins, crop relevant events
            if self.fixedLength:
                frame_transform = transforms.Compose([
                    transforms.Downsample(time_factor=0.001),    #us to ms
                    transforms.TimeAlignment(),         #1st event at t=0
                    transforms.CropTime(max=self.sampleLength),  #crop events
                    transforms.ToFrame(                 #events -> frames
                        sensor_size = (128, 128, 2),
                        time_window=self.samplingTime,  #in ms
                        )
                    ])
            else:  #variable length
                frame_transform = transforms.Compose([
                    transforms.Downsample(time_factor=0.001),  #us to ms
                    transforms.TimeAlignment(),                #1st event at t=0
                    transforms.ToFrame(                        #events -> frames
                        sensor_size = (128, 128, 2),
                        time_window=self.samplingTime,  #in ms
                        )
                    ])
            
            #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
            frames = frame_transform(events)
            
            #input spikes need to be float Tensors reshaped to CHWT for SLAYER
            T, C, W, H = frames.shape
            input_spikes = torch.Tensor(frames).reshape(C, H, W, T) #torch.float32
            
            #if fixedLength, assure sample has nTimeBins (or pad with zeros)
            if self.fixedLength:
                if input_spikes.shape[-1] < self.nTimeBins:
                    padding = torch.zeros(
                        (2, 128, 128, self.nTimeBins - input_spikes.shape[-1]))  
                    input_spikes = torch.cat([input_spikes, padding], dim=-1)

            #choice of binning mode
            """
            By default, Tonic sets the number of spikes at each pixel for every
            time bin as an integer number. SLAYER uses values up to '1.0' in
            the Tensors, and therefore, we can use 2 types of binning mode.
            -type "OR": if there is either 1 OR more spikes at a specific 
                        [x,y] pixel at the same time bin, we set its value 
                        fixed at "1.0 / dt";
            -type "SUM": if there is 1 OR more spikes at a specific [x,y] pixel
                         at the same time bin, we set a value proportional to 
                         the number of spikes, and limited to '1.0'.
            """
            if self.binMode == 'OR' :
                #set all pixels with spikes to the value '1.0/dt'
                input_spikes = torch.where(
                    (input_spikes > 0),   #if spike:
                    1.0 / self.samplingTime,            #set pixel value
                    input_spikes)                       #else keep value 0
            elif self.binMode == 'SUM' :
                input_spikes = torch.where(
                    (input_spikes > 0),   #if spike:
                    input_spikes / input_spikes.max(),  #set pixel value
                    input_spikes)                       #else keep value 0
            else:
                print("Invalid binning mode; results are compromised!")
                print("(binning_mode should be only 'OR' or 'SUM')")
        
        else:
            print("Invalid transform method from events to Tensor of spikes!")
            print("('transf_method' should be 'SlayerTD' or 'TonicFrames')")
        
        return input_spikes, desired_class
    

# Dataset definition
class AnimalsDvsDataset(Dataset):
    """
    SL-Animals-DVS: event-driven sign language animals dataset
    
    Original paper by Ajay Vasudevan, Pablo Negri, Camila Di Ielsi, Bernabe 
    Linares‑Barranco, Teresa Serrano‑Gotarredona.
    """
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 fixedLength, transfMethod, binMode):
        
        self.path = dataPath                            #string
        self.files = np.loadtxt(fileList, dtype='str')  #1D array [max 59]
        self.samplingTime = samplingTime                #5 [ms]
        self.sampleLength = sampleLength                #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime) #300 bins 
        self.fixedLength = fixedLength                  #boolean
        self.transfMethod = transfMethod                #string
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
            Use this method with native Slayer TD (fixed time_bins only)
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
            Use this method with Tonic frames (fixed OR variable time_bins).
            """
            #if using fixed sampleLength/time_bins, crop relevant events
            if self.fixedLength:
                frame_transform = transforms.Compose([
                    transforms.Downsample(time_factor=0.001),    #us to ms
                    transforms.TimeAlignment(),         #1st event at t=0
                    transforms.CropTime(max=self.sampleLength),  #crop events
                    transforms.ToFrame(                 #events -> frames
                        sensor_size = (sensor_shape[0], sensor_shape[1], 2),
                        time_window=self.samplingTime,  #in ms
                        )
                    ])
            else:  #variable length
                frame_transform = transforms.Compose([
                    transforms.Downsample(time_factor=0.001),  #us to ms
                    transforms.TimeAlignment(),                #1st event at t=0
                    transforms.ToFrame(                        #events -> frames
                        sensor_size = (sensor_shape[0], sensor_shape[1], 2),
                        time_window=self.samplingTime,  #in ms
                        )
                    ])
            #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
            frames = frame_transform(events)
            #input spikes need to be float Tensors reshaped to CHWT for SLAYER
            T, C, W, H = frames.shape
            input_spikes = torch.Tensor(frames).reshape(C, H, W, T) #torch.float32
            #if fixedLength, assure sample has nTimeBins (or pad with zeros)
            if self.fixedLength:
                if input_spikes.shape[-1] < self.nTimeBins:
                    padding = torch.zeros(    # shape (2, 128, 128, DiffBins)
                        (2, sensor_shape[0], sensor_shape[1], 
                         self.nTimeBins - input_spikes.shape[-1])
                        )  
                    input_spikes = torch.cat([input_spikes, padding], dim=-1)

            #choice of binning mode
            """
            By default, Tonic sets the number of spikes at each pixel for every
            time bin as an integer number. SLAYER uses values up to '1.0' in
            the Tensors, and therefore, we can use 2 types of binning mode.
            -type "OR": if there is either 1 OR more spikes at a specific 
                        [x,y] pixel at the same time bin, we set its value 
                        fixed at "1.0 / dt";
            -type "SUM": if there is 1 OR more spikes at a specific [x,y] pixel
                         at the same time bin, we set a value proportional to 
                         the number of spikes, and limited to '1.0'.
            """
            if self.binMode == 'OR' :
                #set all pixels with spikes to the value '1.0/dt'
                input_spikes = torch.where(
                    (input_spikes > 0),   #if spike:
                    1.0 / self.samplingTime,            #set pixel value
                    input_spikes)                       #else keep value 0
            elif self.binMode == 'SUM' :
                input_spikes = torch.where(
                    (input_spikes > 0),   #if spike:
                    input_spikes / input_spikes.max(),  #set pixel value
                    input_spikes)                       #else keep value 0
            else:
                print("Invalid binning mode; results are compromised!")
                print("(binning_mode should be only 'OR' or 'SUM')")
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