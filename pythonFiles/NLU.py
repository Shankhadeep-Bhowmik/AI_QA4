import math

# ---------- Activation Functions ----------

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ---------- Input Layer ----------
# Example input (2 features)
inputs = [0.5, 0.8]


# ---------- Hidden Layer ----------
# Weights from Input → Hidden (2 inputs → 2 hidden neurons)
weights_input_hidden = [
    [0.4, 0.7],   # Weights for Hidden Neuron 1
    [0.6, 0.9]    # Weights for Hidden Neuron 2
]

# Bias for hidden neurons
bias_hidden = [0.1, 0.2]

# Compute Hidden Layer Output
hidden_output = []

for i in range(2):  # for each hidden neuron
    weighted_sum = (
        inputs[0] * weights_input_hidden[i][0] +
        inputs[1] * weights_input_hidden[i][1] +
        bias_hidden[i]
    )
    
    activated = relu(weighted_sum)   # Activation function (ReLU)
    hidden_output.append(activated)


# ---------- Output Layer ----------
# Weights from Hidden → Output (2 hidden → 1 output neuron)
weights_hidden_output = [0.8, 0.5]

# Bias for output neuron
bias_output = 0.3

# Compute Final Output
weighted_sum_output = (
    hidden_output[0] * weights_hidden_output[0] +
    hidden_output[1] * weights_hidden_output[1] +
    bias_output
)

final_output = sigmoid(weighted_sum_output)   # Activation (Sigmoid)


# ---------- Display Results ----------

print("Input Layer:", inputs)
print("Hidden Layer Output:", hidden_output)
print("Final Output:", final_output)
