#include <vector>

struct Weight {
    double weight;
    double deltaWeight;
};

class baseNeuron {
public:
    double learningRate = 0.1;
    double momentum = 0.5;
    std::vector<Weight> outputWeights;

    size_t neuronIndex;
    double outputValue;
    double gradient;

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

    double activation(double x) {
        return tanh(x);
    }

    double activationDerivative(double x) {
        return 1.0 - x * x;
    }
};

class Neuron : public baseNeuron {
private:
    std::vector<Weight> recurrentWeights;
    bool isRecurrent;

public:
    Neuron(size_t numOutputs, size_t index, bool recurrent = false) {
        neuronIndex = index;
        isRecurrent = recurrent;

        for (size_t i = 0; i < numOutputs; ++i) {
            outputWeights.push_back(Weight());
            outputWeights.back().weight = randomWeight();
        }
        recurrentWeights.resize(numOutputs);
    }

    void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        gradient = delta * activationDerivative(outputValue);
    }

    void calculateRecurrentGradients(std::vector<double>& previousHiddenLayerOutputs) {
        // Calculate the gradient for this neuron with respect to its output value
        double delta = 0.0;
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            delta += outputWeights[i].weight * outputWeights[i].deltaWeight;
        }

        // Calculate the recurrent gradient based on the delta and previous hidden layer outputs
        gradient = delta * activationDerivative(outputValue);

        // Update the deltas for the output weights (used in weight updates during backpropagation)
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            outputWeights[i].deltaWeight = learningRate * gradient * previousHiddenLayerOutputs[i];
        }
    }

    void feedForward(std::vector<Neuron> currentLayer, std::vector<Neuron> prevLayer) {
        double sum = 0.0;

        for (size_t i = 0; i < currentLayer.size(); ++i) {
            sum += currentLayer[i].getOutputValue() * recurrentWeights[i].weight;
        }

        for (size_t i = 0; i < prevLayer.size(); ++i) {
            sum += prevLayer[i].getOutputValue() * outputWeights[i].weight;
        }

        outputValue = activation(sum);
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
};

class NeuralNetwork {
private:
    std::vector<Neuron> inputLayer;
    std::vector<std::vector<Neuron>> hiddenLayers;
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
            hiddenLayers.push_back(recurrentLayer);
        }

        for (size_t i = 0; i < numOutputs; ++i) {
            outputLayer.push_back(Neuron(numNeuronsPerHiddenLayer, i));
        }
    }

    void getResults(std::vector<double>& resultValues) {
        resultValues.clear();
        for (size_t i = 0; i < outputLayer.size(); ++i) {
            resultValues.push_back(outputLayer[i].getOutputValue());
        }
    }

    void backPropagation(std::vector<std::vector<double>> inputSequence, std::vector<double> targetSequence) {
        if (inputSequence.size() != targetSequence.size()) {
            std::cerr << "Input sequence size does not match target sequence size" << std::endl;
            return;
        }

        // Initialize containers to hold intermediate values for each time step
        std::vector<std::vector<std::vector<double>>> hiddenLayerOutputs(inputSequence.size());
        std::vector<std::vector<double>> outputLayerOutputs(inputSequence.size());

        // Loop through each time step
        for (size_t t = 0; t < inputSequence.size(); ++t) {
            // Perform feedforward pass for each time step
            feedForward(inputSequence[t], inputSequence);

            // Store hidden and output layer outputs for later use in backpropagation
            hiddenLayerOutputs[t].resize(numHiddenLayers);
            for (size_t layer = 0; layer < numHiddenLayers; ++layer) {
                hiddenLayerOutputs[t][layer].resize(hiddenLayers[layer].size());
                for (size_t i = 0; i < hiddenLayers[layer].size(); ++i) {
                    hiddenLayerOutputs[t][layer][i] = hiddenLayers[layer][i].getOutputValue();
                }
            }

            outputLayerOutputs[t].resize(outputLayer.size());
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                outputLayerOutputs[t][i] = outputLayer[i].getOutputValue();
            }

            // Calculate output layer gradients and deltas
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double targetValue = targetSequence[t];
                outputLayer[i].calculateOutputGradients(targetValue);
            }

            // Backpropagate through the recurrent layers
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                for (size_t i = 0; i < hiddenLayers[layer].size(); ++i) {
                    hiddenLayers[layer][i].calculateRecurrentGradients(hiddenLayerOutputs[t][layer]);
                }
            }
        }

        // Update output layer weights
        for (size_t t = 0; t < inputSequence.size(); ++t) {
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                outputLayer[i].updateWeights(hiddenLayers[numHiddenLayers - 1]);
            }
        }

        // Update recurrent layer weights
        for (size_t t = 0; t < inputSequence.size(); ++t) {
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                for (size_t i = 0; i < hiddenLayers[layer].size(); ++i) {
                    hiddenLayers[layer][i].updateWeights((layer == 0) ? inputLayer : hiddenLayers[layer - 1]);
                }
            }
        }
    }

    void feedForward(const std::vector<double>& inputValues, const std::vector<std::vector<double>>& inputSequence) {
        if (inputValues.size() != inputLayer.size()) {
            std::cerr << "Input values size does not match input layer size" << std::endl;
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
                std::vector<Neuron> currentLayer = hiddenLayers[layer];
                for (size_t i = 0; i < numNeuronsPerHiddenLayer; ++i) {
                    // TODO: Check that the resulted output value is correct, may be the same for each neuron in the layer
                    currentLayer[i].feedForward(currentLayer, (layer == 0) ? inputLayer : hiddenLayers[layer - 1]);
                }
            }

            // Perform feedforward for the output layer
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < numNeuronsPerHiddenLayer; ++j) {
                    sum += hiddenLayers[numHiddenLayers - 1][j].getOutputValue() * outputLayer[i].outputWeights[j].weight;
                }
                outputLayer[i].setOutputValue(outputLayer[i].activation(sum));
            }
        }
    }
};