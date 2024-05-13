from random import seed
from random import random
from csv import reader
from math import exp, tanh
import matplotlib.pyplot as plt
import pandas as pd

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation: sigmoid function or sigmoid' according to derivate arg
def transfer_sigmoid(x, derivate):
    if derivate == 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return x * (1.0 - x)
    
def transfer_tanh(x, derivate):
    if derivate == 0:
        return tanh(x)
    else:
        return 1.0 - tanh(x)**2

def forward_propagate(network, row, transfer):
    # first input is set by the dataset array
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, 0)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def predict(network, row, transfer):
    outputs = forward_propagate(network, row, transfer)
    return outputs

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

filename = 'ann_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_inputs = len(dataset[0]) - 1
n_outputs = 1  # Assuming single output

network = initialize_network(n_inputs, 2, n_outputs)

# Predict using the network for all rows in the dataset
predictions = []
for row in dataset:
    normalized_input = [(row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) for i in range(len(row)-1)]
    prediction = predict(network, normalized_input, transfer_tanh)
    predictions.append(prediction[0])  # Assuming single output
file = 'ann_dataset.csv'
f_n=pd.read_csv(file)
print(f_n.head(5))
# Plotting the predicted values
plt.plot(predictions)
# plt.plot(f_n.iloc[:, -1])
plt.title('Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.grid(True)
plt.show()


