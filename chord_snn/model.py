import numpy as np
import sys
import os
import neuro
import risp
import random
import time
import argparse


import json

from encode import Chromagram
from encode import Encoder
from data_loader import DataLoader
import eons 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

"""
Neural Network Training and Evaluation Script for Music Classification

This script provides a mechanism for training a neural network using the EONS evolutionary algorithm on chromagram data extracted from music files. The goal is to classify music into different genres or categories based on their chromagram representations.

Modules:
    - numpy: For numerical operations and data manipulation.
    - sys, os: For system-level operations and file handling.
    - neuro: For neural network operations.
    - risp: For RISP processor operations.
    - random, time: For random seeding and time tracking.
    - argparse: For command-line argument parsing.
    - json: For reading and writing JSON files.
    - encode: For encoding music data into chromagrams.
    - data_loader: For loading the dataset.
    - eons: For the EONS evolutionary algorithm.
    - sklearn: For data splitting, one-hot encoding, and accuracy calculation.

Functions:
    - load_config(filename): Loads a configuration from a JSON file.
    - read_network(fn): Reads a neural network from a JSON file.
    - generate_template_network(properties, n_inputs, n_outputs, seed): Generates a template neural network.
    - get_prediction(proc, x): Gets the predicted label for a given chromagram.
    - fitness(proc, net, X, y): Evaluates the fitness of a network based on its accuracy on a dataset.
    - save_net(net, filename): Saves a neural network to a JSON file.

Usage:
    Run the script with optional command-line arguments:
        --generations: Specifies the number of generations for training (default is 100).
        --network_json: Specifies the path to an existing network JSON file for further training.

    Example:
        python script_name.py --generations 150 --network_json path/to/network.json

Note:
    Ensure that the necessary configuration files (e.g., 'risp.json', 'eons.json') are available in the script's directory or provide the correct paths.

Author:
    [Your Name]
    [Your Email]
    [Any other contact information or credits]
"""


def load_config(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {filename}: {e}")
        sys.exit(1)

risp_config = load_config('/home/dofo/Repos/Github/neuromorphicsnn/chord_snn/config/risp.json')
eo_params = load_config('/home/dofo/Repos/Github/neuromorphicsnn/chord_snn/config/eons.json')


loader_instance = DataLoader(folders= ["minor_triad", "major_triad" ])

# First, split the data into training+validation and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(loader_instance.chroma_files, loader_instance.numerical_label_types, test_size=0.2, random_state=42)
# Now, split the training+validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
# One-hot encode the labels
OneHot = OneHotEncoder(sparse_output=False)
y_train_encoded = OneHot.fit_transform(np.array(y_train).reshape(-1, 1))
y_val_encoded = OneHot.transform(np.array(y_val).reshape(-1, 1))
y_test_encoded = OneHot.transform(np.array(y_test).reshape(-1, 1))

n_inputs = 12
n_hidden = 500
n_outputs = len(OneHot.categories_[0])
n_neurons = n_inputs + n_hidden + n_outputs
n_synapses = 1000


moa = neuro.MOA()
moa.seed(42)

proc = risp.Processor(risp_config) # RISP processor

def generate_template_network(n_inputs, n_outputs, seed=0):
    net = neuro.Network()
    net.set_properties(proc.get_network_properties())
    
    moa = neuro.MOA()
    moa.seed(seed)

    for i in range(n_inputs):
        node = net.add_node(i)
        net.add_input(i)
        net.randomize_node_properties(moa, node)

    for i in range(n_outputs):
        node = net.add_node(i+n_inputs)
        net.add_output(i+n_inputs)
        net.randomize_node_properties(moa, node)

    return net

def read_network(fn):
    try:
        with open(fn, 'r') as f:
            s = f.read()
            j = json.loads(s)
            net = neuro.Network()
            net.from_json(j)
        return net
    except Exception as e:
        print(f"Error reading network from {fn}: {e}")
        sys.exit(1)

parser = argparse.ArgumentParser(description='Neural Network Training Script')
parser.add_argument('--generations', type=int, default=100, help='Number of generations for training')
parser.add_argument('--network_json', type=str, default=None, help='Path to the network JSON file')
parser.add_argument('--patience', type=int, default=10, help='Number of generations without improvement before stopping')

args = parser.parse_args()

max_generations = args.generations

evolver = eons.EONS(eo_params)

if args.network_json:
    network = read_network(args.network_json)
    evolver.set_template_network(network)
else:
    evolver.set_template_network(generate_template_network(n_inputs, n_outputs))

pop = evolver.generate_population(eo_params,42)

def get_prediction(proc, x):
    # Load the chroma data from the npy file
    encoder=Encoder(x)
    
    proc.clear_activity()
    proc.apply_spikes(encoder.spikes)
    proc.run(encoder.time_steps * encoder.num_frames * 4) # you might adjust this duration based on your needs
    
    # Decoding the output to get the predicted label. You might need to adjust this
    predicted_index = proc.output_count_max(n_outputs)[0]
    # Convert index to one-hot encoded format
    one_hot_prediction = np.zeros(n_outputs)
    one_hot_prediction[predicted_index] = 1.0
    return one_hot_prediction 

def fitness(proc, net, X, y):
    proc.load_network(net)
    
    # Set up output tracking
    for i in range(n_outputs):
        proc.track_neuron_events(i)
    
    y_predict = [get_prediction(proc, x) for x in X]
    return accuracy_score(y_predict, y)

vals = []

best_val_accuracy = 0
generations_without_improvement = 0


for generation in range(max_generations):
    start_time = time.time()  # Record the start time for this generation

    # Evaluate fitness of all networks in the population
    fitnesses = [fitness(proc, net.network, X_train, y_train_encoded) for net in pop.networks]
    
    # Track and print best fitness in the current generation
    best_fitness = max(fitnesses)
    vals.append(best_fitness)
    best_net = pop.networks[fitnesses.index(best_fitness)].network

    end_time = time.time()  # Record the end time for this generation
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}, Time: {elapsed_time}, Max_Nodes: {best_net.num_nodes()}")

     # Get predictions for the validation set
    val_accuracy = fitness(proc,best_net, X_val, y_val_encoded)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        generations_without_improvement = 0
    else:
        generations_without_improvement += 1

    if generations_without_improvement >= args.patience:
        print(f"Early stopping after {generation + 1} generations due to no improvement in validation accuracy.")
        break

    # Produce the next generation based on the current population's fitness
    pop = evolver.do_epoch(pop, fitnesses, eo_params)

# Optionally, you can evaluate and print the performance of the best network on the test set here.

def save_net(net,filename):
        try:
            with open(filename, 'w') as f:
                json.dump(net.to_json(), f)
        except Exception as e:
            print(f"Error saving network to {filename}: {e}")

save_net(best_net.as_json().to_python(), 'best_net.json')

print("Evolution finished!")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")

# Get predictions for the test set
y_test_predict = [get_prediction(proc, x) for x in X_test]

# Calculate the accuracy on the test set
test_accuracy = accuracy_score(y_test_encoded, y_test_predict)
print(f"Accuracy on the validation set: {test_accuracy:.4f}")