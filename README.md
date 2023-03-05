# SL-animals-DVS-slayer
This repository contains a SLAYER (Spiking Layer Error Reassignment in Time) implementation on the SL-Animals-DVS dataset using Pytorch and the SLAYER package software.

**A BRIEF INTRODUCTION:**  
SLAYER is an offline training method that directly trains a SNN.  
Therefore, it is a suitable method to train SNNs, which are biologically plausible networks (in short).  
The SL-animals-DVS is a dataset of sign language (SL) gestures peformed by different people representing animals, and recorded with a Dynamic Vision Sensor (DVS).  

<p align="center">
<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs10044-021-01011-w/MediaObjects/10044_2021_1011_Fig4_HTML.png" width="300px></p>

<p align="center"> </p>  

The reported results in the SL-animals paper were a test accuracy of 60.9% +- 4.58% in the whole dataset and 78.03% +- 3.08% on the reduced dataset (excluding group S3). The results achieved with the implementation published here: **Test Accuracy (whole dataset): 54.83% +- 6.61%; Test Accuracy (exluding S3): 62.05% +- 5.78%**.  
           
## Requirements
While not sure if the list below contain the actual minimums, it will run for sure if you do have the following:
- Python 3.0+
- Pytorch 1.11+
- CUDA 11.3+
- slayerSNN (installation instructions [here](https://github.com/bamsumit/slayerPytorch))
- python libraries: os, numpy, matplotlib, pandas, sklearn, datetime, tonic, pyyaml, h5py, tensorboardX

## README FIRST
This package contains the necessary python files to train a Spiking Neural Network with the SLAYER method on the Sign Language Animals DVS dataset. 

**IMPLEMENTATION**  
Package Contents:  
- dataset.py
- learning_tools.py
- model.py
- network.yaml
- sl_animals_slayer.py
- train_test_only.py
- utils.py

The SL-Animals-DVS dataset implementation code is in *dataset.py*, and it's basically a Pytorch Dataset object. The library [*Tonic*](https://tonic.readthedocs.io/en/latest/index.html#) was used to read and process the DVS recordings.  

The main functions to train, test, split the dataset and plot the results are in *learning_tools.py*. The Spiking Neural Network model is in *model.py* (MyNetwork), and reproduces the architecture described in the SL-animals paper. The file *network.yaml* contains the main parameters that can be customized like batch size, sampling time, simulation window, neuron type, data path, and many others.  

 The main program is in *sl_animals_slayer.py*, which uses the correct experimental procedure for training a network using cross validation after dividing the dataset into train, validation and test sets. A simpler version of the main program is in *train_test_only.py*, which is basically the same except dividing the dataset only into train and test sets, in an effort to replicate the published results. Apparently, the benchmark results were reported in this simpler dataset split configuration, which is not optimal.  

 Finally, *utils.py* contains some functions to visualize the dataset samples, and split the dataset recordings into slices and saving it to disk.

## Use
1. Clone this repository:
```
git clone https://github.com/ronichester/SL-animals-DVS-slayer
```
2. Download the dataset in [this link](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database);
3. Save the DVS recordings in the *data/recordings* folder and the file tags in the *data/tags* folder;
4. Edit the custom parameters according to your preferences in *network.yaml*. The default parameters setting is functional and was tailored according to the information provided in the relevant papers, the reference codes used as a basis, and mostly by trial and error (lots of it!). You are encouraged to edit the main parameters and **let me know if you got better results**.
6. Run *sl_animals_slayer.py* (or *train_test_only.py*) to start the SNN training:
```
python sl_animals_slayer.py
```
or
```
python train_test_only.py
```
7. The network weights, training curves and spike plots will be saved in *src/output*. The Tensorboard logs will be saved in *src/logs*; to visualize the training with tensorboard:
  - open a terminal (I use Anaconda Prompt), go to the *src* directory and type:
```
tensorboard --logdir=logs
```
  - open your browser and type in the address bar http://localhost:6006/ or any other address shown in the terminal screen.
  

## References 
- Vasudevan, A., Negri, P., Di Ielsi, C. et al. ["SL-Animals-DVS: event-driven sign language animals dataset"](https://doi.org/10.1007/s10044-021-01011-w) . *Pattern Analysis and Applications 25, 505–520 (2021)*. 
- Shrestha SB, Orchard G; [SLAYER: spike layer error reassignment in time](https://arxiv.org/pdf/1810.08646); *Advances in neural information processing systems, pp 1412–1421 (2018)*
- The original dataset can be downloaded [here](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database)

## Copyright
Copyright 2023 Schechter Roni. This software is free to use, copy, modify and distribute for personal, academic, or research use. Its terms are described under the General Public License, GNU v3.0.
