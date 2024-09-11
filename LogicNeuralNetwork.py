import numpy as np

# Define the sigmoid activation function: https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function (used in backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights randomly with mean 0
input_size = 2 # Two input nodes become one output node after processing in the hidden layers
hidden_size = 4 # When i originally had the size of the neurons as 2, the network was unable to learn the XOR function for [1, 1] but increasing it's size to 4
output_size = 1 # One output node because the output of any logic is either 0 or 1

weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

def forward_propagation(input_data):
    # Input to hidden layer - 
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Hidden to output layer
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backpropagation(input_data, target, learning_rate):
    global weights_input_hidden, weights_hidden_output
    
    # Perform forward propagation
    hidden_layer_output, output_layer_output = forward_propagation(input_data)

    # Calculate error
    output_error = target - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    # Backpropagate the error to the hidden layer
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate

# XOR training data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # All possible inputs
# This is the output of the training that the algorithm compares to...thus one of these needs to always be selected.
target_output = np.array([[0], [1], [1], [1]]) # OR target
# target_output = np.array([[0], [1], [1], [0]]) # XOR target
# target_output = np.array([[1], [0], [0], [0]]) # NOR target
# target_output = np.array([[1], [0], [0], [1]]) # XNOR target
# target_output = np.array([[0], [0], [0], [1]]) # AND target
# target_output = np.array([[1], [1], [1], [0]]) # NAND target

epochs = 100000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    backpropagation(input_data, target_output, learning_rate)

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        _, predicted_output = forward_propagation(input_data)
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(target_output - predicted_output))}')

def print_predictions(predictions):    
    for i, prediction in enumerate(predictions):
        print(f"Input {i+1}: Prediction = {prediction[0]:.4f} => {'1' if prediction[0] > 0.5 else '0'}")

# Test the network on custom inputs
new_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
_, predicted_output = forward_propagation(new_inputs)

print("Predictions after training (formatted):")
print_predictions(predicted_output)
