import configparser
from ast import literal_eval
import numpy as np

data = configparser.ConfigParser()
data.read("data.ini")

# Input dataset
indata = np.array(literal_eval(data["Data"]["inputdata"]))

# Desired output dataset
outdata = np.array(literal_eval(data["Data"]["outputdata"]))

# When to print the current progress
checkups = {10000 * i for i in range(1,11)}

# Seed random numbers for added replicability
np.random.seed(1)

# Defining the sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# The derivative of the sigmoid function
def sig_deriv(x):
    return x*(1-x)

# Define "synapses". Initialize weights randomly
# Over a cont distrib with half open interval -1 to 1
# synapse0 dimensions: input points per dataset, number of datasets
# synapse1 dimensions: number of neurons below number of neurons above
synapses0 = 2*np.random.random((3,4)) - 1
synapses1 = 2*np.random.random((4,1)) - 1

for iter in range(100000):
    # Forward propagation
    layer0 = indata
    layer1 = sigmoid(np.dot(layer0,synapses0))
    layer2 = sigmoid(np.dot(layer1,synapses1))
    # How much did we miss by?
    l2_error = outdata - layer2
    if iter in checkups:
        print("Error now:" + str(np.mean(np.abs(l2_error))))
    #use sigmoid to get an adjustment value for l2
    l2_delta = l2_error * sig_deriv(layer2)
    #get l1 errors based on contribution to l2 errors
    l1_error = l2_delta.dot(synapses1.T)
    #get l1 deltas
    l1_delta = l1_error * sig_deriv(layer1)
    #update weights of neurons
    synapses1 += layer1.T.dot(l2_delta)
    synapses0 += layer0.T.dot(l1_delta)

print("Testing the net:")
cont = 1
while cont:
    new_situation = literal_eval(input("Please Enter a Situation > "))
    newlay1 = sigmoid(np.dot(new_situation, synapses0))
    output = float(sigmoid(np.dot(newlay1,synapses1)))
    prediction = bool(round(output))
    conf = (0.5 + abs(0.5 - output)) * 100
    print("Prediction: {}, {:.2f} percent confident".format(prediction, conf))
