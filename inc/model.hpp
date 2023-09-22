#include <vector>
#include <iostream>
#include <cmath>

struct Weight {
    double weight;
    double deltaWeight;
};

class Neuron {
private:
    Weight biasWeight;
    Weight recurrentWeight;

    bool isRecurrent;

    double learningRate = 0.1;
    double momentum = 0.5;

    double outputValue;
    double gradient;

public:
    Neuron(size_t numOutputs, bool recurrent = false) {
        isRecurrent = recurrent;

        biasWeight.weight = randomWeight();
        biasWeight.deltaWeight = 0.0;
    }

    double randomWeight() {
        double out = rand() / double(RAND_MAX);
        // std::cout << "Random weight: " << out << std::endl;
        return out;
    }

    bool getIsRecurrent() {
        return isRecurrent;
    }

    void setOutputValue(double val) {
        outputValue = val;
    }

    double getOutputValue() {
        return outputValue;
    }

    Weight& getBiasWeights() {
        return biasWeight;
    }

    Weight& getRecurrentWeights() {
        return recurrentWeight;
    }

    double activationFunction(double x) {
        return tanh(x);
    }

    double activationFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    double getGradient() {
        return gradient;
    }

    void setGradient(double val) {
        gradient = val;
    }

    double getLearningRate() {
        return learningRate;
    }

    double calculateDelta(Neuron& neuron) {
        double oldDeltaWeight = neuron.biasWeight.deltaWeight;
        double newDeltaWeight = learningRate * neuron.getOutputValue() * gradient + momentum * oldDeltaWeight;
        return newDeltaWeight;
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
            neurons.emplace_back(numOutputsPerNeuron, recurrent);
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
            double delta = neurons[i].calculateDelta(prevNeurons[i]);

            if (neurons[i].getIsRecurrent()) {
                prevNeurons[i].getRecurrentWeights().deltaWeight = delta;
                prevNeurons[i].getRecurrentWeights().weight += delta;
            }
            else {
                prevNeurons[i].getBiasWeights().deltaWeight = delta;
                prevNeurons[i].getBiasWeights().weight += delta;
            }
        }
    }

    void calculateRecurrentGradients(std::vector<double> previousHiddenLayerOutputs) {
        // Calculate the gradient for this neuron with respect to its output value
        double delta = 0.0;
        for (size_t i = 0; i < neurons.size(); ++i) {
            delta += neurons[i].getRecurrentWeights().weight * neurons[i].getRecurrentWeights().deltaWeight;
        }

        for (size_t i = 0; i < neurons.size(); ++i) {
            // Update the neuron's gradient
            neurons[i].setOutputValue(previousHiddenLayerOutputs[i]);
            neurons[i].setGradient(delta * neurons[i].activationFunctionDerivative(neurons[i].getOutputValue()));

            // Update the deltas for the output weights (used in weight updates during backpropagation)
            neurons[i].getRecurrentWeights().deltaWeight = neurons[i].getLearningRate() * neurons[i].getGradient() * previousHiddenLayerOutputs[i];
        }
    }

    void feedForward(Layer& prevLayer) {
        std::vector<Neuron>& prevNeurons = prevLayer.getVector();
        for (size_t i = 0; i < neurons.size(); ++i) {
            // neurons[i].feedForward(neurons, prevNeurons);
            double delta = 0.0;

            for (size_t j = 0; j < neurons.size(); ++j) {
                delta += neurons[j].getOutputValue() * neurons[j].getRecurrentWeights().weight;
            }

            for (size_t j = 0; j < prevNeurons.size(); ++j) {
                delta += prevNeurons[j].getOutputValue() * neurons[j].getBiasWeights().weight;
            }

            neurons[i].setOutputValue(neurons[i].activationFunction(delta));
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
                layers[layer].calculateRecurrentGradients(layers[layer-1].getOutputs());
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
                    delta += layers[numHiddenLayers][j].getOutputValue() * outputLayer[i].getBiasWeights().weight;
                }
                outputLayer[i].setOutputValue(outputLayer[i].activationFunction(delta));
            }
        }
    }
};