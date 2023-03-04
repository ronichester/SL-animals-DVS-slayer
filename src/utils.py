# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:39:43 2022

@author: Schechter
"""
#import libraries
import os
import tonic
import numpy as np
import pandas as pd
import slayerSNN as snn
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tonic.transforms as transforms


#visualize dataset sample on animation
def animate_events(dataset, sample_index, time_window=50, frame_rate=24, repeat=False):
    """
    Generates an animation on a dataset sample. The sample is retrieved
    as a 1D array of events in the Tonic format (x, y, t, p), in [us] 
    resolution.
    
        dataset: torch Dataset object
        sample_index: int, must be between [0, 1120]
        time_window: int, time window in ms for each frame (default 50 ms)
        frame_rate: int, (default 24 FPS)
        repeat: bool, loop the animation infinetely (default is False)
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
       
    #create transform object
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size,                 # bin into frames
                           time_window=time_window)     # in [ms]
        ])  
    
    #transform event array -> frames (shape TCWH (time_bins, 2, 128, 128))
    frames = frame_transform(sample_events)
    
    #interval between frames in ms (default=41.6)
    interval = 1e3 / frame_rate  # in ms
    
    #defining 1st frame: image is the difference between polarities
    fig = plt.figure()
    plt.title("class name: {}".format(sample_class))
    image = plt.imshow((frames[0][1]-frames[0][0]).T, cmap='gray')
    
    #update the data on each frame
    def animate(frame):
        image.set_data((frame[1]-frame[0]).T)  
        return image

    animation = anim.FuncAnimation(fig, animate, frames=frames, 
                                   interval=interval, repeat=repeat)
    plt.show()
    
    return animation


#visualize dataset sample on plots (by time bins)
def plot_events(dataset, sample_index):
    """
    Generates a plot with 3 frames on a dataset sample. The events of a sample
    are divided in 3 time bins, each frame accumulates the events of 1 bin.
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
    
    #transform event array -> frames (shape TCWH)
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size, n_time_bins=3)  # events -> 3 frames
        ])  
    frames = frame_transform(sample_events)

    def plot_frames(frames):
        fig, axes = plt.subplots(1, len(frames))
        fig.suptitle("class name: {}".format(sample_class))
        for axis, frame in zip(axes, frames):
            axis.imshow((frame[1] - frame[0]).T, cmap='gray')
            axis.axis("off")
        plt.tight_layout()
        plt.show()
    
    plot_frames(frames)
    
    return frames


def events_to_TD(events, invert_xy=False):
    """
    Basically transforms a structured 1D array of events (of type='object') 
    into the Slayer event object (TD). 
    
    The original Tonic output format (output of tonic.io.read_dvs_128) is a 
    structured 1D array, a sequence of events where each event is a 
    tuple like object with 4 elements (x, y, t, p):
        x - pixel x index [0 - 127]
        y - pixel y index [0 - 127]
        t - timestamp (in [us])
        p - polarity or channel (True or False)
    
    This dataset (SL animals) was recorded in us resolution and with a time 
    offset (first event is far away from timestamp 0).
    
    To work in SLAYER:
        1) time resolution should be in [ms]
        2) timestamp shoud start at t=0
    
    A flag 'invert_xy' was added to transpose 'x' and 'y' axes. 
    When using Tonic to read these dataset files, the axes are inverted.
    """
    #transform events resolution and remove time offset
    events_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),  # us to ms
        transforms.TimeAlignment(),                # 1st event starts at t=0
        ])
    events = events_transform(events)
    
    #create the Slayer event object (TD)
    x, y = (events['y'], events['x']) if invert_xy else (events['x'], events['y'])
    p, t = events['p'], events['t']
    TD = snn.io.event(x, y, p, t)
   
    return TD


def animate_TD(dataset, sample_index):
    #get sample events
    sample_events = dataset.get_sample(sample_index)[0]
    #visualize events animation
    inTD = events_to_TD(sample_events, True)  
    animation = snn.io.animTD(inTD)
    
    return animation


def list_sliced_files(raw_file_list):
    #create a list of sliced files, given a list of 'raw' recording files
    sliced_file_list = []
    for file in raw_file_list:
        for i in range(19):
            sliced_file_list.append(file + '_{}.npy'.format(str(i).zfill(2)))
    
    return sliced_file_list


def slice_data(data_path, file_list):
    """
    A script to slice the SL animals DVS recordings into actual samples for 
    training, and save the slices to disk. 
    
    Parameters:
        data_path: str;
            The 'raw' data path, for the 59 dvs recordings
        file_list: txt file;
            A text file with a list of the 'raw' file names
    """
    print('Checking for sliced dataset:')
    
    #create sliced dataset directory and path
    os.makedirs(data_path + "sliced_recordings", exist_ok=True)
    sliced_data_path = data_path + "sliced_recordings/"
    
    #load file names into a 1D array
    files = np.loadtxt(file_list, dtype='str')  #1D array [max 59]
    
    #check if dataset is already sliced
    if len(os.listdir(sliced_data_path)) < (19 * len(files)):
        
        print('Slicing the dataset, this may take a while...')
        
        #For each of the raw recordings: slice in 19 pieces and save to disk
        for record_name in files:
            print('Processing record {}...'.format(record_name))
            
            #read the DVS file
            """
            The output of this function:
                sensor_shape: tuple, the DVS resolution (128, 128)
                events: 1D-array, the sequential events on the file
                        1 microsecond resolution
                        each event is 4D and has the shape 'xytp'
            """
            sensor_shape, events = tonic.io.read_dvs_128(data_path + 'recordings/'
                                                         + record_name + '.aedat')
            #read the tag file
            tagfile = pd.read_csv(data_path + 'tags/' + record_name + '.csv')  #df
            
            #define event boundaries for each class
            events_start = list(tagfile["startTime_ev"])
            events_end = list(tagfile["endTime_ev"])
            
            #create a list of arrays, separating the recording in 19 slices
            sliced_events = tonic.slicers.slice_events_at_indices(events, 
                                                                 events_start, 
                                                                 events_end)
            #save 19 separate events on disk
            for i, chosen_slice in enumerate(sliced_events):
                np.save(sliced_data_path + '{}_{}.npy'.format(
                    record_name, str(i).zfill(2)), chosen_slice)
        print('Slicing completed.\n')
        
    else:
        print('Dataset is already sliced.\n')

    # import glob
    # #create a list of the sliced dataset files
    # sliced_data_list = glob.glob1(sliced_data_path, '*.npy')
                


