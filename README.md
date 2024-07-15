# SL-animals-DVS training with SLAYER
This repository contains a SLAYER (Spiking Layer Error Reassignment in Time) implementation on the SL-Animals-DVS dataset using Pytorch and the SLAYER package software.

**A BRIEF INTRODUCTION:**  
SLAYER is an offline training method that directly trains a Spiking Neural Network (SNN). 
Therefore, it is a suitable method to train SNNs, which are biologically plausible networks (in short).  
The SL-animals-DVS is a dataset of sign language (SL) gestures peformed by different people representing animals, and recorded with a Dynamic Vision Sensor (DVS).  

<p align="center">
<img src="https://github.com/ronichester/SL-animals-DVS-slayer/blob/main/samples_and_outputs/SL_animals_sample2.gif" width="600px"></p>

<p align="center">
<img src="https://github.com/ronichester/SL-animals-DVS-slayer/blob/main/samples_and_outputs/SL_animals_sample10.gif" width="600px"></p>

<p align="center"> </p>  

The reported results in the SL-animals paper were divided in two: results with the full dataset and results with a reduced dataset, meaning excluding group S3. The results achieved with the implementation published here fall short of the published results, but get fairly close, considering the published results have no code available to reproduce them.  
  
**The implementation published in this repository is the first publicly available SLAYER implementation on the SL-animals dataset** (and the only one as of may 2023, as far as I know). The results are summarized below:

|       | Full Dataset | Reduced Dataset |
|:-:|:-:|:-:|
| Reported Results    | 60.9 +- 4.58 % | 78.03 +- 3.08 % |
| This Implementation | 54.3 +- 6.14 % | 61.41 +- 3.28 % |

           
## Requirements
While not sure if the list below contains the actual minimums, it will run for sure if you do have the following:
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

The main functions to train, test, split the dataset and plot the results are in *learning_tools.py*. The Spiking Neural Network model is in *model.py* (MyNetwork), and reproduces the architecture described in the SL-animals paper. The file *network.yaml* contains the main parameters that can be customized like *batch size*, *sampling time*, *simulation window*, *neuron type*, *data path*, and many others.  

A new feature was introduced as an option, allowing the use of random sample crops for training instead of the fixed crops starting at the beggining of the sample, as in the original paper implementation. This allows further exploration of the available data in the dataset.

 The main program is in *sl_animals_slayer.py*, which uses the correct experimental procedure for training a network using cross validation after dividing the dataset into train, validation and test sets. A simpler version of the main program is in *train_test_only.py*, which is basically the same except dividing the dataset only into train and test sets, in an effort to replicate the published results. Apparently, the benchmark results were reported in this simpler dataset split configuration.  

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
