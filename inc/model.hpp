#include <vector>
#include <iostream>
#include <cmath>
#include <random>

class LSTM {
private:
    int inputSize;
    int hiddenSize;

    std::vector<std::vector<float>> weightsInput;
    std::vector<std::vector<float>> weightsHidden;
    std::vector<float> bias;

    std::vector<float> cellState;
    std::vector<float> hiddenState;

public:
    LSTM(int inputSize, int hiddenSize)
        : inputSize(inputSize), hiddenSize(hiddenSize) {
        // Initialize weights and biases
        weightsInput.resize(4, std::vector<float>(inputSize, 0.0));
        weightsHidden.resize(4, std::vector<float>(hiddenSize, 0.0));
        bias.resize(4, 0.0);

        // Initialize cell state and hidden state
        cellState.resize(hiddenSize, 0.0);
        hiddenState.resize(hiddenSize, 0.0);
    }

    void forward(const std::vector<float>& input) {
        // LSTM forward pass
        std::vector<float> concatInput(inputSize + hiddenSize, 0.0);
        for (int i = 0; i < inputSize; ++i) {
            concatInput[i] = input[i];
        }
        for (int i = 0; i < hiddenSize; ++i) {
            concatInput[inputSize + i] = hiddenState[i];
        }

        std::vector<float> gates(4, 0.0);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < inputSize + hiddenSize; ++j) {
                gates[i] += weightsInput[i][j] * concatInput[j];
            }
            gates[i] += bias[i];
            gates[i] = sigmoid(gates[i]);
        }

        // Element-wise operations for updating cell state and hidden state
        for (int i = 0; i < hiddenSize; ++i) {
            cellState[i] = cellState[i] * gates[2] + gates[0] * gates[1];
            hiddenState[i] = tanh(cellState[i]) * gates[3];
        }
    }

    std::vector<float> getHiddenState() const {
        return hiddenState;
    }

private:
    float sigmoid(float x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    float tanh(float x) {
        return std::tanh(x);
    }
};

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

    // Recurrent layer parameters
    int sequenceLength;
    std::vector<std::vector<float>> recurrentWeights;

    std::vector<LSTM> lstmLayers;

public:
    RecurrentNeuralNetwork(int inputSize, int hiddenLayerSize, int outputLayerSize, int numEpochs, float learningRate, int sequenceLength)
        : inputSize(inputSize), hiddenLayerSize(hiddenLayerSize), outputLayerSize(outputLayerSize), numEpochs(numEpochs), learningRate(learningRate), sequenceLength(sequenceLength) {
        // Initialize weights and layers
        lstmLayers.resize(sequenceLength, LSTM(inputSize, hiddenLayerSize));

        weightsInputHidden.resize(inputSize, std::vector<float>(hiddenLayerSize, 0.0));
        weightsHiddenOutput.resize(hiddenLayerSize, std::vector<float>(outputLayerSize, 0.0));
        hiddenLayer.resize(sequenceLength, std::vector<float>(hiddenLayerSize, 0.0));
        outputLayer.resize(outputLayerSize, 0.0);

        // Initialize recurrent layer parameters
        recurrentWeights.resize(hiddenLayerSize, std::vector<float>(hiddenLayerSize, 0.0));
    }

    void fit(const std::vector<std::vector<float>>& inputData, const std::vector<int>& targetData) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.1, 0.1);

        // Randomly initialize weights
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenLayerSize; ++j) {
                weightsInputHidden[i][j] = dist(gen);
            }
        }

        for (int i = 0; i < hiddenLayerSize; ++i) {
            for (int j = 0; j < outputLayerSize; ++j) {
                weightsHiddenOutput[i][j] = dist(gen);
            }
            for (int j = 0; j < hiddenLayerSize; ++j) {
                recurrentWeights[i][j] = dist(gen);
            }
        }

        // Training loop
        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            for (size_t example = 0; example < inputData.size(); ++example) {
                // Forward pass
                std::vector<float> input = inputData[example];
                int target = targetData[example];

                // Initialize hidden state for this sequence
                for (int t = 0; t < sequenceLength; ++t) {
                    if (t == 0) {
                        hiddenLayer[t] = forward(input);
                    } else {
                        hiddenLayer[t] = forward(hiddenLayer[t - 1]);
                    }
                }

                // Final output prediction
                std::vector<float> predicted = hiddenLayer[sequenceLength - 1];

                // Backpropagation
                backward(input, target, predicted);
            }

            std::cout << "Epoch: " << epoch + 1 << " / " << numEpochs << std::endl;
        }
    }

    std::vector<float> predict(std::vector<float>& input) {
        // Implement the forward pass for prediction here
        return forward(input);
    }

private:
    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // ReLU activation function
    float relu(float x) {
        return std::max(0.0f, x);
    }

    // Forward pass
    std::vector<float> forward(std::vector<float>& input) {
        std::vector<float> currentHidden(hiddenLayerSize, 0.0);

        int t; // Declare t here

        // Forward pass using LSTM layers
        for (t = 0; t < sequenceLength; ++t) {
            lstmLayers[t].forward(input);
            input = lstmLayers[t].getHiddenState();
        }

        t = sequenceLength - 1;

        // Now you can use t in the following calculations

        // Compute hidden layer activations
        for (int i = 0; i < hiddenLayerSize; ++i) {
            float activation = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                activation += input[j] * weightsInputHidden[j][i];
            }
            for (int j = 0; j < hiddenLayerSize; ++j) {
                activation += hiddenLayer[t - 1][j] * recurrentWeights[j][i];
            }
            currentHidden[i] = sigmoid(activation);
        }

        // Compute output layer activations
        for (int i = 0; i < outputLayerSize; ++i) {
            float activation = 0.0;
            for (int j = 0; j < hiddenLayerSize; ++j) {
                activation += currentHidden[j] * weightsHiddenOutput[j][i];
            }
            outputLayer[i] = sigmoid(activation);
        }

        // Update hidden layer state
        hiddenLayer.push_back(currentHidden);
        hiddenLayer.erase(hiddenLayer.begin());

        return outputLayer;
    }

    // Backpropagation
    void backward(const std::vector<float>& input, const int& target, const std::vector<float>& predicted) {
        // Compute output layer errors and deltas
        std::vector<float> outputErrors(outputLayerSize);
        std::vector<float> outputDeltas(outputLayerSize);

        for (int i = 0; i < outputLayerSize; ++i) {
            outputErrors[i] = target - predicted[i];
            outputDeltas[i] = outputErrors[i] * predicted[i] * (1 - predicted[i]);
        }

        // Update hidden-to-output weights
        for (int i = 0; i < hiddenLayerSize; ++i) {
            for (int j = 0; j < outputLayerSize; ++j) {
                weightsHiddenOutput[i][j] += learningRate * hiddenLayer[sequenceLength - 1][i] * outputDeltas[j];
            }
        }

        // Compute hidden layer errors and deltas
        std::vector<float> hiddenErrors(hiddenLayerSize);
        std::vector<float> hiddenDeltas(hiddenLayerSize);

        for (int i = 0; i < hiddenLayerSize; ++i) {
            hiddenErrors[i] = 0.0;
            for (int t = sequenceLength - 1; t >= 0; --t) {
                for (int j = 0; j < outputLayerSize; ++j) {
                    hiddenErrors[i] += outputDeltas[j] * weightsHiddenOutput[i][j];
                }
                hiddenDeltas[i] = hiddenErrors[i] * hiddenLayer[t][i] * (1 - hiddenLayer[t][i]);

                // Update recurrent weights
                for (int k = 0; k < hiddenLayerSize; ++k) {
                    recurrentWeights[k][i] += learningRate * hiddenLayer[t][k] * hiddenDeltas[i];
                }

                // Update input-to-hidden weights for the current time step
                for (int j = 0; j < inputSize; ++j) {
                    weightsInputHidden[j][i] += learningRate * input[j] * hiddenDeltas[i];
                }

                // Propagate error to the previous time step
                if (t > 0) {
                    for (int k = 0; k < hiddenLayerSize; ++k) {
                        hiddenErrors[k] += hiddenDeltas[i] * recurrentWeights[i][k];
                    }
                }
            }
        }
    }
};