# chord_snn

## Description

chord_snn is a project that implements a spiking neural network (SNN) for chord recognition. It uses a biologically-inspired approach to model the behavior of neurons and their connections. It uses TENNLabs to simulate a neuroprocessor. Training is done using an Evolutonary Optimization Algorithm. For access to both TENNLabs framework and the Evolutionary Optimization Algorithm, please contact cschuman@vols.utk.edu.

## Installation

1. Clone the repository

2. Install the required dependencies:

Make sure you have Python 3.6 or later installed.  
Install the required packages using pip.

midi2audio
mido
librosa

Other dependencies are included in the framework requirements no need to worry!


## Usage

1. Use piano_chord_gen.ipynb to generate the dataset. Otherwise use your own dataset as long as it is in .wav format.
2. Model.py constructs and trains the model, while data_loader.py loads and organizes the dataset. econde.py encodes the data into spikes. 

Note: Only the python files are needed to run the model. The jupyter notebooks are for demonstration purposed and debugging purposes. Bewarned, they are messy!

Author: Adolfo Partida