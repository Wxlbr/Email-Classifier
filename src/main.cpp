#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class RecurrentNeuralNetwork {
private:
    int inputSize;
    int hiddenLayerSize;
    int outputLayerSize;
    int numEpochs;
    float learningRate;

    std::vector<std::vector<float>> weightsInputHidden;
    std::vector<std::vector<float>> weightsHiddenOutput;
    std::vector<std::vector<float>> hiddenLayer;
    std::vector<float> outputLayer;

public:
    RecurrentNeuralNetwork(int inputSize, int hiddenLayerSize, int numEpochs, float learningRate)
        : inputSize(inputSize), hiddenLayerSize(hiddenLayerSize), numEpochs(numEpochs), learningRate(learningRate) {
        // outputLayerSize = 1;  // Set outputLayerSize to 1
        // Initialize weights and layers
        weightsInputHidden.resize(inputSize, std::vector<float>(hiddenLayerSize, 0.0));
        weightsHiddenOutput.resize(hiddenLayerSize, std::vector<float>(outputLayerSize, 0.0));
        hiddenLayer.resize(hiddenLayerSize, std::vector<float>(1, 0.0));
        outputLayer.resize(outputLayerSize, 0.0);
    }

    void fit(const std::vector<std::vector<float>>& inputData, const std::vector<std::vector<float>>& targetData) {
        // Training loop
        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            for (size_t example = 0; example < inputData.size(); ++example) {
                // Forward pass
                std::vector<float> input = inputData[example];
                std::vector<float> target = targetData[example];

                // Forward pass
                std::vector<float> predicted = forward(input);

                // Backpropagation
                backward(input, target, predicted);
            }
        }
    }

    std::vector<float> predict(const std::vector<float>& input) {
        // Implement the forward pass for prediction here
        std::vector<float> predictions = forward(input);
        return predictions;
    }

private:
    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // relu activation function
    float relu(float x) {
        return std::max(0.0f, x);
    }

    // Forward pass
    std::vector<float> forward(const std::vector<float>& input) {
        // Compute hidden layer activations
        for (int i = 0; i < hiddenLayerSize; ++i) {
            float activation = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                activation += input[j] * weightsInputHidden[j][i];
            }
            hiddenLayer[i][0] = sigmoid(activation);
        }

        // Compute output layer activations
        for (int i = 0; i < outputLayerSize; ++i) {
            float activation = 0.0;
            for (int j = 0; j < hiddenLayerSize; ++j) {
                activation += hiddenLayer[j][0] * weightsHiddenOutput[j][i];
            }
            outputLayer[i] = sigmoid(activation);
        }

        return outputLayer;
    }

    // Backpropagation
    void backward(const std::vector<float>& input, const std::vector<float>& target, const std::vector<float>& predicted) {
        // Compute output layer errors and deltas
        std::vector<float> outputErrors(outputLayerSize);
        std::vector<float> outputDeltas(outputLayerSize);

        for (int i = 0; i < outputLayerSize; ++i) {
            outputErrors[i] = target[i] - predicted[i];
            outputDeltas[i] = outputErrors[i] * predicted[i] * (1 - predicted[i]);
        }

        // Update hidden-to-output weights
        for (int i = 0; i < hiddenLayerSize; ++i) {
            for (int j = 0; j < outputLayerSize; ++j) {
                weightsHiddenOutput[i][j] += learningRate * hiddenLayer[i][0] * outputDeltas[j];
            }
        }

        // Compute hidden layer errors and deltas
        std::vector<float> hiddenErrors(hiddenLayerSize);
        std::vector<float> hiddenDeltas(hiddenLayerSize);

        for (int i = 0; i < hiddenLayerSize; ++i) {
            hiddenErrors[i] = 0.0;
            for (int j = 0; j < outputLayerSize; ++j) {
                hiddenErrors[i] += outputDeltas[j] * weightsHiddenOutput[i][j];
            }
            hiddenDeltas[i] = hiddenErrors[i] * hiddenLayer[i][0] * (1 - hiddenLayer[i][0]);
        }

        // Update input-to-hidden weights
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenLayerSize; ++j) {
                weightsInputHidden[i][j] += learningRate * input[i] * hiddenDeltas[j];
            }
        }
    }
};

int main() {
    std::cout << "Simple RNN Model Implementation" << std::endl;

    // Define RNN parameters
    int input_size = 2;
    int hidden_size = 100;
    int num_epochs = 1000; // Increased the number of epochs for better training
    float learning_rate = 0.01; // Increased the learning rate

    // Create a simple RNN model
    RecurrentNeuralNetwork model(input_size, hidden_size, num_epochs, learning_rate);

    // Create example input and target data
    std::vector<std::vector<float>> input_data = {
        {0.1, 0.2, 0.3, 0.4},
        {0.5, 0.6},
        {0.7, 0.8, 0.9, 1.0, 1.1, 1.2}
    };
    std::vector<std::vector<float>> target_data = {
        {0.0},
        {0.0},
        {1.0}
    };

    // Train the model
    model.fit(input_data, target_data);

    std::cout << "Training complete!" << std::endl;

    // Make predictions
    std::vector<float> input_example = {0.7, 0.8, 1.2};
    std::vector<float> predictions = model.predict(input_example);

    // Print predictions
    std::cout << "Predictions: ";
    for (float pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    return 0;
}
