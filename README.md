# Neural Network Implementation for Logical Functions

This repository contains a simple implementation of a neural network to learn and predict logical functions using the sigmoid activation function. The neural network is designed to handle various logical operations like XOR, OR, NOR, XNOR, AND, and NAND.

## Code Overview

### Functions

- **`sigmoid(x)`**

  - Computes the sigmoid activation function.
  - [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)

- **`sigmoid_derivative(x)`**

  - Computes the derivative of the sigmoid function, used during backpropagation.

- **`forward_propagation(input_data)`**

  - Performs the forward pass of the neural network.
  - Returns the output of the hidden layer and the final output layer.

- **`backpropagation(input_data, target, learning_rate)`**

  - Performs backpropagation to update weights based on the error between predicted and target values.

- **`print_predictions(predictions)`**
  - Prints the network's predictions formatted as binary values (0 or 1).

### Hyperparameters

- **`input_size`**: Number of input nodes (2)
- **`hidden_size`**: Number of neurons in the hidden layer (4)
- **`output_size`**: Number of output nodes (1)
- **`epochs`**: Number of iterations for training (100,000)
- **`learning_rate`**: Rate at which weights are updated (0.1)

### Training Data

- **XOR Training Data**:
    ```python
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_output = np.array([[0], [1], [1], [0]])
    ```

Uncomment the desired `target_output` for different logical functions:

- **OR**: `target_output = np.array([[0], [1], [1], [1]])`
- **NOR**: `target_output = np.array([[1], [0], [0], [0]])`
- **XNOR**: `target_output = np.array([[1], [0], [0], [1]])`
- **AND**: `target_output = np.array([[0], [0], [0], [1]])`
- **NAND**: `target_output = np.array([[1], [1], [1], [0]])`

### Running the Code

1. Clone the repository.
2. Ensure you have `numpy` installed (`pip install numpy`).
3. Run the script to train the neural network and see the results.

    ```bash
    python neural_network.py
    ```

### Output

The script prints the error at every 1000 epochs during training and shows the final predictions after training for the logical functions. The predictions are formatted as binary values indicating the output of the logical function for each input.

#### Example Output

    ```bash
    Epoch 0, Error: 0.4867
    ...
    Epoch 100000, Error: 0.0000
    Predictions after training (formatted):
    Input 1: Prediction = 0.0000 => 0
    Input 2: Prediction = 0.9999 => 1
    Input 3: Prediction = 0.9998 => 1
    Input 4: Prediction = 0.0001 => 0
    ```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
