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

    double activationFunction(double x) {
        return tanh(x);
    }

    double activationFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    double calculateDelta(Neuron& neuron) {
        double oldDeltaWeight = neuron.outputWeights[neuronIndex].deltaWeight;
        double newDeltaWeight = learningRate * neuron.getOutputValue() * gradient + momentum * oldDeltaWeight;
        return newDeltaWeight;
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

    void updateWeights(std::vector<Neuron>& prevLayer) {
        for (size_t i = 0; i < prevLayer.size(); ++i) {
            Neuron& neuron = prevLayer[i];

            double delta = calculateDelta(neuron);

            if (isRecurrent) {
                recurrentConnections[neuronIndex].deltaWeight = delta;
                recurrentConnections[neuronIndex].weight += delta;
            }
            else {
                neuron.outputWeights[neuronIndex].deltaWeight = delta;
                neuron.outputWeights[neuronIndex].weight += delta;
            }
        }
    }

    void calculateRecurrentGradients(std::vector<double> previousHiddenLayerOutputs) {
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

class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(size_t numNeurons, size_t numOutputsPerNeuron, bool recurrent = false) {
        for (size_t i = 0; i < numNeurons; ++i) {
            neurons.emplace_back(numOutputsPerNeuron, i, recurrent);
        }
    }

    Neuron& operator[](size_t index) {
        return neurons[index];
    }

    size_t size() {
        return neurons.size();
    }

    std::vector<Neuron>& getVector() {
        return neurons;
    }

    std::vector<double> getOutputs() {
        std::vector<double> outputValues;
        for (size_t n = 0; n < neurons.size(); ++n) {
            outputValues.push_back(neurons[n].getOutputValue());
        }
        return outputValues;
    }

    void updateWeights(Layer& prevLayer) {
        std::vector<Neuron>& prevNeurons = prevLayer.getVector();
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].updateWeights(prevNeurons);
        }
    }

    void feedForward(Layer& prevLayer) {
        std::vector<Neuron>& prevNeurons = prevLayer.getVector();
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].feedForward(neurons, prevNeurons);
        }
    }
};

class NeuralNetwork {
private:
    std::vector<Layer> layers;

    int numInputs;
    int numOutputs;
    int numHiddenLayers;
    int numNeuronsPerHiddenLayer;

public:
    NeuralNetwork(int numInputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, int numOutputs = 1) :
        numInputs(numInputs), numHiddenLayers(numHiddenLayers), numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), numOutputs(numOutputs) {

        // Input layer
        layers.emplace_back(numInputs, numNeuronsPerHiddenLayer);

        // Hidden layers
        for (size_t i = 0; i < numHiddenLayers; ++i) {
            layers.emplace_back(numNeuronsPerHiddenLayer, numNeuronsPerHiddenLayer, true);
        }

        // Output layer
        layers.emplace_back(numOutputs, numNeuronsPerHiddenLayer);
    }

    std::vector<double> getResults() {
        return layers.back().getOutputs();
    }

    void backPropagation(std::vector<std::vector<double>> inputSequence, std::vector<int> targetSequence) {
        size_t inputSize = inputSequence.size();
        size_t targetSize = targetSequence.size();

        if (inputSize != targetSize) {
            std::cerr << "inputSequence.size() != targetSequence.size()" << std::endl;
            return;
        }

        // Loop through each time step
        for (size_t t = 0; t < inputSize; ++t) {
            // Perform feedforward pass for each time step
            feedForward(inputSequence[t], inputSequence);

            // Calculate output layer gradients and deltas
            Layer& outputLayer = layers.back();
            for (size_t i = 0; i < numOutputs; ++i) {
                double targetValue = targetSequence[t];
                outputLayer[i].calculateOutputGradients(targetValue);
            }

            // Backpropagate through the recurrent layers
            for (int layer = numHiddenLayers; layer >= 1; --layer) {
                for (size_t i = 0; i < layers[layer].size(); ++i) {
                    layers[layer][i].calculateRecurrentGradients(layers[layer-1].getOutputs());
                }
            }
        }

        // Update output layer weights
        Layer& outputLayer = layers.back();
        for (size_t t = 0; t < inputSize; ++t) {
            outputLayer.updateWeights(layers[numHiddenLayers]);
        }

        // Update recurrent layer weights
        for (size_t t = 0; t < inputSize; ++t) {
            for (int layer = numHiddenLayers; layer >= 1; --layer) {
                layers[layer].updateWeights(layers[layer - 1]);
            }
        }
    }

    void feedForward(const std::vector<double>& inputValues, const std::vector<std::vector<double>>& inputSequence) {
        Layer& inputLayer = layers[0];
        Layer& outputLayer = layers.back();

        if (inputValues.size() != inputLayer.size()) {
            std::cerr << "inputValues.size() != inputLayer.size()" << std::endl;
            return;
        }

        // Assign input values to input neurons
        for (size_t i = 0; i < inputValues.size(); ++i) {
            inputLayer[i].setOutputValue(inputValues[i]);
        }

        // Define the number of time steps to perform feedforward
        // TODO: Consider using the size of the input sequence
        size_t step = 1; // inputSequence.size();

        // Perform feedforward for each time step
        for (size_t t = 0; t < step; ++t) {
            // Perform feedforward for each hidden layer
            for (size_t layer = 1; layer <= numHiddenLayers; ++layer) {
                // Perform feedforward for each neuron in the hidden layer
                for (size_t i = 0; i < layers[layer].size(); ++i) {
                    layers[layer].feedForward((layer == 1) ? inputLayer : layers[layer - 1]);
                }
            }

            // Perform feedforward for the output layer
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double delta = 0.0;
                for (size_t j = 0; j < layers[numHiddenLayers].size(); ++j) {
                    delta += layers[numHiddenLayers][j].getOutputValue() * outputLayer[i].getOutputWeights()[j].weight;
                }
                outputLayer[i].setOutputValue(outputLayer[i].activationFunction(delta));
            }
        }
    }
};