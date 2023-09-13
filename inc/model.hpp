#include <vector>
#include <cassert>

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
};

class Neuron;

class RecurrentNeuron : public baseNeuron {
private:
    std::vector<Weight> recurrentConnections;

public:
    RecurrentNeuron(size_t numOutputs, size_t index) {
        neuronIndex = index;
        for (size_t c = 0; c < numOutputs; ++c) {
            outputWeights.push_back(Weight());
            outputWeights.back().weight = randomWeight();
        }

        recurrentConnections.resize(numOutputs);
    }

    void calculateRecurrentGradients(std::vector<double>& previousHiddenLayerOutputs) {
        // Calculate the gradient for this neuron with respect to its output value
        double delta = 0.0;
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            delta += outputWeights[i].weight * outputWeights[i].deltaWeight;
        }

        // Calculate the recurrent gradient based on the delta and previous hidden layer outputs
        double recurrentGradient = delta * activationFunctionDerivative(outputValue);

        // Update the neuron's gradient
        gradient = recurrentGradient;

        // Update the deltas for the output weights (used in weight updates during backpropagation)
        for (size_t i = 0; i < outputWeights.size(); ++i) {
            outputWeights[i].deltaWeight = learningRate * recurrentGradient * previousHiddenLayerOutputs[i];
        }
    }

    void feedForward(std::vector<RecurrentNeuron> prevRecurrentLayer, std::vector<Neuron> prevLayer) {
        double sum = 0.0;

        for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
            sum += prevRecurrentLayer[n].getOutputValue() * recurrentConnections[n].weight;
        }

        for (size_t n = 0; n < prevLayer.size(); ++n) {
            sum += prevLayer[n].getOutputValue() * outputWeights[n].weight;
        }

        outputValue = activationFunction(sum);
    }

    void feedForward(std::vector<RecurrentNeuron> prevRecurrentLayer, std::vector<RecurrentNeuron> prevLayer) {
        double sum = 0.0;

        for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
            sum += prevRecurrentLayer[n].getOutputValue() * recurrentConnections[n].weight;
        }

        for (size_t n = 0; n < prevLayer.size(); ++n) {
            sum += prevLayer[n].getOutputValue() * outputWeights[n].weight;
        }

        outputValue = activationFunction(sum);
    }

    void updateWeights(std::vector<Neuron>& prevLayer) {
        for (size_t n = 0; n < prevLayer.size(); ++n) {
            Neuron& prevNeuron = prevLayer[n];

            double oldDeltaWeight = prevNeuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * prevNeuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            prevNeuron.outputWeights[n].deltaWeight = newDeltaWeight;
            prevNeuron.outputWeights[n].weight += newDeltaWeight;
        }
    }

    void updateRecurrentWeights(std::vector<RecurrentNeuron>& prevRecurrentLayer) {
        for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
            RecurrentNeuron& prevRecurrentNeuron = prevRecurrentLayer[n];

            double oldDeltaWeight = prevRecurrentNeuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * prevRecurrentNeuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            recurrentConnections[n].deltaWeight = newDeltaWeight;
            recurrentConnections[n].weight += newDeltaWeight;
        }
    }

    void calculateRecurrentGradient(std::vector<RecurrentNeuron>& nextRecurrentLayer) {
        double sum = 0.0;
        for (size_t n = 0; n < nextRecurrentLayer.size(); ++n) {
            sum += nextRecurrentLayer[n].outputWeights[neuronIndex].weight * nextRecurrentLayer[n].gradient;
        }

        gradient = sum * (1.0 - outputValue * outputValue);
    }
};

class Neuron : public baseNeuron {
public:
    Neuron(size_t numOutputs, size_t index) {
        neuronIndex = index;
        for (size_t c = 0; c < numOutputs; ++c) {
            outputWeights.push_back(Weight());
            outputWeights.back().weight = randomWeight();
        }
    }

    void updateInputWeights(std::vector<RecurrentNeuron>& prevLayer) {
        for (size_t n = 0; n < prevLayer.size(); ++n) {
            RecurrentNeuron& neuron = prevLayer[n];

            double oldDeltaWeight = neuron.outputWeights[neuronIndex].deltaWeight;
            double newDeltaWeight = learningRate * neuron.getOutputValue() * gradient + momentum * oldDeltaWeight;

            neuron.outputWeights[neuronIndex].deltaWeight = newDeltaWeight;
            neuron.outputWeights[neuronIndex].weight += newDeltaWeight;
        }
    }

    void calculateHiddenGradients(std::vector<RecurrentNeuron>& nextLayer) {
        double delta = 0.0;
        for (size_t n = 0; n < nextLayer.size() - 1; ++n) {
            delta += outputWeights[n].weight * nextLayer[n].gradient;
        }

        gradient = delta * (1.0 - outputValue * outputValue);
    }

    void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        gradient = delta * (1.0 - outputValue * outputValue);
    }
};

class NeuralNetwork {
private:
    std::vector<Neuron> inputLayer;
    std::vector<std::vector<RecurrentNeuron>> recurrentLayers;
    std::vector<RecurrentNeuron> recurrentLayer;
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
            recurrentLayer.clear();
            for (size_t j = 0; j < numNeuronsPerHiddenLayer; ++j) {
                recurrentLayer.push_back(RecurrentNeuron(numNeuronsPerHiddenLayer, j));
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
        assert(inputSequence.size() == targetSequence.size());

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
                hiddenLayerOutputs[t][layer].resize(recurrentLayers[layer].size());
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    hiddenLayerOutputs[t][layer][i] = recurrentLayers[layer][i].getOutputValue();
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
                for (size_t i = 0; i < recurrentLayers[layer].size(); ++i) {
                    recurrentLayers[layer][i].calculateRecurrentGradients(hiddenLayerOutputs[t][layer]);
                }
            }
        }

        // Update output layer weights
        for (size_t t = 0; t < inputSequence.size(); ++t) {
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                outputLayer[i].updateInputWeights(recurrentLayers[numHiddenLayers - 1]);
            }
        }

        // Update recurrent layer weights
        for (size_t t = 0; t < inputSequence.size(); ++t) {
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
        // std::cout << "inputValues.size(): " << inputValues.size() << std::endl;
        // std::cout << "inputLayer.size(): " << inputLayer.size() << std::endl;

        assert(inputValues.size() == inputLayer.size());

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
                    // recurrentLayers[layer][i].feedForward(recurrentLayers[layer], (layer == 0) ? inputLayer : recurrentLayers[layer - 1]);
                    if (layer == 0) {
                        recurrentLayers[layer][i].feedForward(recurrentLayers[layer], inputLayer);
                    }
                    else {
                        recurrentLayers[layer][i].feedForward(recurrentLayers[layer], recurrentLayers[layer - 1]);
                    }
                }
            }

            // Perform feedforward for the output layer
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < recurrentLayers[numHiddenLayers - 1].size(); ++j) {
                    sum += recurrentLayers[numHiddenLayers - 1][j].getOutputValue() * outputLayer[i].outputWeights[j].weight;
                }
                outputLayer[i].setOutputValue(outputLayer[i].activationFunction(sum));
            }
        }
    }
};