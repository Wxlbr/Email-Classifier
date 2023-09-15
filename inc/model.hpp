#include <vector>
#include <iostream>

struct Weight {
    double weight;
    double deltaWeight;
};

class Neuron {
private:
    std::vector<Weight> recurrentConnections;
    bool isRecurrent;

    double learningRate = 0.1;
    double momentum = 0.5;
    std::vector<Weight> outputWeights;

    size_t neuronIndex;
    double outputValue;
    double gradient;

public:
    Neuron(size_t numOutputs, size_t index, bool recurrent = false) {
        neuronIndex = index;
        isRecurrent = recurrent;

        for (size_t c = 0; c < numOutputs; ++c) {
            outputWeights.push_back(Weight());
            outputWeights.back().weight = randomWeight();
        }

        recurrentConnections.resize(numOutputs);
    }

    // TODO: Consider use of std::random_device to seed a RNG
    double randomWeight() {
        return rand() / double(RAND_MAX);
    }

    void setOutputValue(double val) {
        outputValue = val;
    }

    double getOutputValue() {
        return outputValue;
    }

    std::vector<Weight>& getOutputWeights() {
        return outputWeights;
    }

    double getLearningRate() {
        return learningRate;
    }

    double getMomentum() {
        return momentum;
    }

    double activationFunction(double x) {
        return tanh(x);
    }

    double activationFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    void feedForward(std::vector<Neuron> prevRecurrentLayer, std::vector<Neuron> prevLayer) {
        double delta = 0.0;

        for (size_t i = 0; i < prevRecurrentLayer.size(); ++i) {
            delta += prevRecurrentLayer[i].getOutputValue() * recurrentConnections[i].weight;
        }

        for (size_t i = 0; i < prevLayer.size(); ++i) {
            delta += prevLayer[i].getOutputValue() * outputWeights[i].weight;
        }

        outputValue = activationFunction(delta);
    }

    void updateInputWeights(std::vector<Neuron>& prevLayer) {
        for (size_t i = 0; i < prevLayer.size(); ++i) {
            Neuron& neuron = prevLayer[i];

            double oldDeltaWeight = neuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * neuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            neuron.outputWeights[neuronIndex].deltaWeight = newDeltaWeight;
            neuron.outputWeights[neuronIndex].weight += newDeltaWeight;
        }
    }

    void updateWeights(std::vector<Neuron>& prevLayer) {
        for (size_t i = 0; i < prevLayer.size(); ++i) {
            Neuron& prevNeuron = prevLayer[i];

            double oldDeltaWeight = prevNeuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * prevNeuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            prevNeuron.outputWeights[i].deltaWeight = newDeltaWeight;
            prevNeuron.outputWeights[i].weight += newDeltaWeight;
        }
    }

    void updateRecurrentWeights(std::vector<Neuron>& prevRecurrentLayer) {
        for (size_t i = 0; i < prevRecurrentLayer.size(); ++i) {
            Neuron& prevRecurrentNeuron = prevRecurrentLayer[i];

            double oldDeltaWeight = prevRecurrentNeuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * prevRecurrentNeuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            recurrentConnections[i].deltaWeight = newDeltaWeight;
            recurrentConnections[i].weight += newDeltaWeight;
        }
    }

    void calculateRecurrentGradients(std::vector<double>& previousHiddenLayerOutputs) {
        // Calculate the gradient for this neuron with respect to its output value
        double delta = 0.0;
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            delta += outputWeights[i].weight * outputWeights[i].deltaWeight;
        }

        // Update the neuron's gradient
        gradient = delta * activationFunctionDerivative(outputValue);

        // Update the deltas for the output weights (used in weight updates during backpropagation)
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            outputWeights[i].deltaWeight = learningRate * gradient * previousHiddenLayerOutputs[i];
        }
    }

    void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        gradient = delta * activationFunctionDerivative(outputValue);
    }
};

class NeuralNetwork {
private:
    std::vector<Neuron> inputLayer;
    std::vector<std::vector<Neuron>> recurrentLayers;
    std::vector<Neuron> outputLayer;

    int numInputs;
    int numOutputs;
    int numHiddenLayers;
    int numNeuronsPerHiddenLayer;

public:
    NeuralNetwork(int numInputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, int numOutputs = 1) :
        numInputs(numInputs), numHiddenLayers(numHiddenLayers), numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), numOutputs(numOutputs) {

        for (size_t i = 0; i < numInputs; ++i) {
            inputLayer.push_back(Neuron(numNeuronsPerHiddenLayer, i));
        }

        for (size_t i = 0; i < numHiddenLayers; ++i) {
            std::vector<Neuron> recurrentLayer;
            for (size_t j = 0; j < numNeuronsPerHiddenLayer; ++j) {
                recurrentLayer.push_back(Neuron(numNeuronsPerHiddenLayer, j, true));
            }
            recurrentLayers.push_back(recurrentLayer);
        }

        for (size_t i = 0; i < numOutputs; ++i) {
            outputLayer.push_back(Neuron(numNeuronsPerHiddenLayer, i));
        }
    }

    void getResults(std::vector<double>& resultValues) {
        resultValues.clear();
        for (size_t n = 0; n < outputLayer.size(); ++n) {
            resultValues.push_back(outputLayer[n].getOutputValue());
        }
    }

    void backPropagation(std::vector<std::vector<double>> inputSequence, std::vector<double> targetSequence) {
        size_t inputSize = inputSequence.size();
        size_t targetSize = targetSequence.size();

        if (inputSize != targetSize) {
            std::cerr << "inputSequence.size() != targetSequence.size()" << std::endl;
            return;
        }

        // Initialize containers to hold intermediate values for each time step
        std::vector<std::vector<std::vector<double>>> hiddenLayerOutputs(inputSize);
        std::vector<std::vector<double>> outputLayerOutputs(inputSize);

        // Loop through each time step
        for (size_t t = 0; t < inputSize; ++t) {
            // Perform feedforward pass for each time step
            feedForward(inputSequence[t], inputSequence);

            // Store hidden and output layer outputs for later use in backpropagation
            hiddenLayerOutputs[t].resize(numHiddenLayers);
            for (size_t layer = 0; layer < numHiddenLayers; ++layer) {
                hiddenLayerOutputs[t][layer].resize(recurrentLayers[layer].size());
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    hiddenLayerOutputs[t][layer][i] = recurrentLayers[layer][i].getOutputValue();
                }
            }

            outputLayerOutputs[t].resize(numOutputs);
            for (size_t i = 0; i < numOutputs; ++i) {
                outputLayerOutputs[t][i] = outputLayer[i].getOutputValue();
            }

            // Calculate output layer gradients and deltas
            for (size_t i = 0; i < numOutputs; ++i) {
                double targetValue = targetSequence[t];
                outputLayer[i].calculateOutputGradients(targetValue);
            }

            // Backpropagate through the recurrent layers
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    recurrentLayers[layer][i].calculateRecurrentGradients(hiddenLayerOutputs[t][layer]);
                }
            }
        }

        // Update output layer weights
        for (size_t t = 0; t < inputSize; ++t) {
            for (size_t i = 0; i < numOutputs; ++i) {
                outputLayer[i].updateInputWeights(recurrentLayers[numHiddenLayers - 1]);
            }
        }

        // Update recurrent layer weights
        for (size_t t = 0; t < inputSize; ++t) {
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    // recurrentLayers[layer][i].updateRecurrentWeights(layer == 0 ? inputLayer : hiddenLayerOutputs[t][layer - 1]);
                    if (layer == 0) {
                        recurrentLayers[layer][i].updateWeights(inputLayer);
                    }
                    else {
                        recurrentLayers[layer][i].updateRecurrentWeights(recurrentLayers[layer - 1]);
                    }
                }
            }
        }
    }

    void feedForward(const std::vector<double>& inputValues, const std::vector<std::vector<double>>& inputSequence) {
        if (inputValues.size() != inputLayer.size()) {
            std::cerr << "inputValues.size() != inputLayer.size()" << std::endl;
            return;
        }

        // Assign input values to input neurons
        for (size_t i = 0; i < inputValues.size(); ++i) {
            inputLayer[i].setOutputValue(inputValues[i]);
        }

        // Perform feedforward for each time step
        for (size_t t = 0; t < inputSequence.size(); ++t) {
            // Perform feedforward for each hidden layer
            for (size_t layer = 0; layer < numHiddenLayers; ++layer) {
                // Perform feedforward for each neuron in the hidden layer
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    recurrentLayers[layer][i].feedForward(recurrentLayers[layer], (layer == 0) ? inputLayer : recurrentLayers[layer - 1]);
                }
            }

            // Perform feedforward for the output layer
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double delta = 0.0;
                for (size_t j = 0; j < recurrentLayers[numHiddenLayers - 1].size(); ++j) {
                    delta += recurrentLayers[numHiddenLayers - 1][j].getOutputValue() * outputLayer[i].getOutputWeights()[j].weight;
                }
                outputLayer[i].setOutputValue(outputLayer[i].activationFunction(delta));
            }
        }
    }
};