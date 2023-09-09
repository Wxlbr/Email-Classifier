#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct RecurrentConnection {
    double weight;
    double deltaWeight;
};

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

// TODO: Consider using a parent node class for RecurrentNeuron and Neuron
class RecurrentNeuron {
public:
    // TODO: Make private and add getters and setters for RecurrentNeuron class
    static double learningRate;
    static double momentum;
    vector<Connection> outputWeights;
    vector<RecurrentConnection> recurrentConnections;

    // TODO: Consider use of std::random_device to seed a RNG
    static double randomWeight() { return rand() / double(RAND_MAX); }

    double outputValue;
    double state; // Internal state for a recurrent neuron
    size_t neuronIndex;
    double gradient;

    void setOutputValue(double val) { outputValue = val; }
    double getOutputValue() { return outputValue; }

    double getLearningRate() { return learningRate; }
    double getMomentum() { return momentum; }

    void calculateRecurrentGradients(const vector<double>& previousHiddenLayerOutputs);

    double activationFunction(double x) {
        return tanh(x);
    }

    double activationFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    void feedForward(std::vector<RecurrentNeuron> prevRecurrentLayer, std::vector<Neuron> prevLayer);
    void feedForwardRecurrent(std::vector<RecurrentNeuron> prevRecurrentLayer, std::vector<RecurrentNeuron> prevLayer);

    void updateWeights(const vector<Neuron>& prevLayer);

    void updateRecurrentWeights(const vector<RecurrentNeuron>& prevRecurrentLayer) {
        for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
            RecurrentNeuron& prevRecurrentNeuron = const_cast<RecurrentNeuron&>(prevRecurrentLayer[n]);
            double oldDeltaWeight = prevRecurrentNeuron.outputWeights[neuronIndex].deltaWeight;

            double newDeltaWeight =
                learningRate
                * prevRecurrentNeuron.getOutputValue()
                * gradient
                + momentum
                * oldDeltaWeight;
            recurrentConnections[n].deltaWeight = newDeltaWeight;
            recurrentConnections[n].weight += newDeltaWeight;
        }
    }

    double calculateRecurrentGradient(const vector<RecurrentNeuron>& nextRecurrentLayer) {
        double sum = 0.0;

        for (size_t n = 0; n < nextRecurrentLayer.size(); ++n) {
            sum += nextRecurrentLayer[n].outputWeights[neuronIndex].weight * nextRecurrentLayer[n].gradient;
        }

        gradient = sum * (1.0 - state * state);
        return gradient;
    }

    RecurrentNeuron(size_t numOutputs, size_t index) : neuronIndex(index) {
        for (size_t c = 0; c < numOutputs; ++c) {
            outputWeights.push_back(Connection());
            outputWeights.back().weight = randomWeight();
        }

        recurrentConnections.resize(numOutputs);
    }
};

class Neuron {
public:
    // TODO: Make private and add getters and setters for Neuron class
    static double learningRate;
    static double momentum;
    vector<Connection> outputWeights;

    // TODO: Consider use of std::random_device to seed a RNG
    static double randomWeight() { return rand() / double(RAND_MAX); }

    double outputValue;
    size_t neuronIndex;
    double gradient;

    void setOutputValue(double val) { outputValue = val; }
    double getOutputValue() const { return outputValue; }

    void updateInputWeights(vector<RecurrentNeuron>& prevLayer) {
        for (size_t n = 0; n < prevLayer.size(); ++n) {
            RecurrentNeuron& neuron = prevLayer[n];
            double oldDeltaWeight = neuron.outputWeights[neuronIndex].deltaWeight;

            double newDeltaWeight =
                learningRate
                * neuron.getOutputValue()
                * gradient
                + momentum
                * oldDeltaWeight;
            neuron.outputWeights[neuronIndex].deltaWeight = newDeltaWeight;
            neuron.outputWeights[neuronIndex].weight += newDeltaWeight;
        }
    }

    double sumDOW(const vector<RecurrentNeuron>& nextLayer) const {
        double sum = 0.0;

        for (size_t n = 0; n < nextLayer.size() - 1; ++n) {
            sum += outputWeights[n].weight * nextLayer[n].gradient;
        }

        return sum;
    }

    void calculateHiddenGradients(const vector<RecurrentNeuron>& nextLayer) {
        double dow = sumDOW(nextLayer);
        gradient = dow * (1.0 - outputValue * outputValue);
    }

    void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        gradient = delta * (1.0 - outputValue * outputValue);
    }

    double activationFunction(double x) {
        return tanh(x);
    }

    void feedForward(vector<RecurrentNeuron> prevLayer) {
        double sum = 0.0;

        for (size_t n = 0; n < prevLayer.size(); ++n) {
            sum += prevLayer[n].getOutputValue() *
                prevLayer[n].outputWeights[neuronIndex].weight;
        }

        outputValue = activationFunction(sum);
    }

    void feedForward(double sum) {
        outputValue = activationFunction(sum);
    }

    Neuron(size_t numOutputs, size_t index) : neuronIndex(index) {
        for (size_t c = 0; c < numOutputs; ++c) {
            outputWeights.push_back(Connection());
            outputWeights.back().weight = randomWeight();
        }
    }
};

void RecurrentNeuron::feedForward(vector<RecurrentNeuron> prevRecurrentLayer, vector<Neuron> prevLayer) {
    double sum = 0.0;

    for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
        sum += prevRecurrentLayer[n].getOutputValue() * recurrentConnections[n].weight;
    }

    for (size_t n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputValue() * outputWeights[n].weight;
    }

    state = tanh(sum);
    outputValue = state;
}

void RecurrentNeuron::feedForwardRecurrent(vector<RecurrentNeuron> prevRecurrentLayer, vector<RecurrentNeuron> prevLayer) {
    double sum = 0.0;

    for (size_t n = 0; n < prevRecurrentLayer.size(); ++n) {
        sum += prevRecurrentLayer[n].getOutputValue() * recurrentConnections[n].weight;
    }

    for (size_t n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputValue() * outputWeights[n].weight;
    }

    state = tanh(sum);
    outputValue = state;
}

void RecurrentNeuron::updateWeights(const vector<Neuron>& prevLayer) {
    for (size_t n = 0; n < prevLayer.size(); ++n) {
        Neuron& prevNeuron = const_cast<Neuron&>(prevLayer[n]);
        double oldDeltaWeight = prevNeuron.outputWeights[neuronIndex].deltaWeight;

        double newDeltaWeight =
            learningRate
            * prevNeuron.getOutputValue()
            * gradient
            + momentum
            * oldDeltaWeight;
        outputWeights[n].deltaWeight = newDeltaWeight;
        outputWeights[n].weight += newDeltaWeight;
    }
}

void RecurrentNeuron::calculateRecurrentGradients(const vector<double>& previousHiddenLayerOutputs) {
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

// TODO: Move assignment
double RecurrentNeuron::learningRate = 0.1;
double RecurrentNeuron::momentum = 0.5;
double Neuron::learningRate = 0.1;
double Neuron::momentum = 0.5;

class NeuralNetwork {
private:
    vector<Neuron> inputLayer;
    vector<vector<RecurrentNeuron>> recurrentLayers;
    vector<RecurrentNeuron> recurrentLayer;
    vector<Neuron> outputLayer;

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

    void getResults(vector<double>& resultValues) const {
        resultValues.clear();
        for (size_t n = 0; n < outputLayer.size(); ++n) {
            resultValues.push_back(outputLayer[n].getOutputValue());
        }
    }

    void backPropagation(vector<vector<double>> inputSequence, vector<double> targetSequence) {
        assert(inputSequence.size() == targetSequence.size());

        // Initialize containers to hold intermediate values for each time step
        vector<vector<vector<double>>> hiddenLayerOutputs(inputSequence.size());
        vector<vector<double>> outputLayerOutputs(inputSequence.size());

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

    void feedForward(const vector<double>& inputValues, const vector<vector<double>>& inputSequence) {
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
                        recurrentLayers[layer][i].feedForwardRecurrent(recurrentLayers[layer], recurrentLayers[layer - 1]);
                    }
                }
            }

            // Perform feedforward for the output layer
            for (size_t i = 0; i < outputLayer.size(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < recurrentLayers[numHiddenLayers - 1].size(); ++j) {
                    sum += recurrentLayers[numHiddenLayers - 1][j].getOutputValue() * outputLayer[i].outputWeights[j].weight;
                }
                outputLayer[i].feedForward(sum);
            }
        }
    }
};

int main() {
    // Load data from emails.csv
    vector<vector<double>> data;
    string flagHeader = "Prediction";
    vector<string> header;
    vector<int> labels;
    ifstream file("./inc/emailsHotEncoding.csv");
    string line;
    getline(file, line);
    stringstream ss(line);
    string cell;
    int flagIndex = -1;
    int i = 0;

    while (getline(ss, cell, ',')) {
        if (cell == flagHeader) {
            flagIndex = i;
        }
        else {
            header.push_back(cell);
        }
        i++;
    }

    if (flagIndex == -1) {
        cerr << "Flag header not found: " << flagHeader << endl;
        return 1;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        int index = 0;

        while (getline(ss, cell, ',')) {
            if (index == flagIndex) {
                labels.push_back(stoi(cell));
            }
            else {
                row.push_back(stod(cell));
            }
            index++;
        }

        row.resize(10); // Shrink row to first x elements
        data.push_back(row);
    }

    data.resize(500); // Shrink data to first x rows

    vector<vector<double>> input_data = data;
    vector<int> target_data = labels;
    // vector<size_t> topology = { 3000, 5, 1 };
    int inputSize = input_data[0].size();

    // numInputs, numHiddenLayers, numNeuronsPerHiddenLayer, numOutputs
    NeuralNetwork myNetwork(inputSize, 1, 5, 1);

    // Train test split, 80% train, 20% test
    int train_size = (int)(input_data.size() * 0.8);
    int test_size = input_data.size() - train_size;
    vector<vector<double>> train_input_data;
    vector<double> train_target_data;
    vector<vector<double>> test_input_data;
    vector<double> test_target_data;

    for (int i = 0; i < train_size; ++i) {
        train_input_data.push_back(input_data[i]);
        train_target_data.push_back(target_data[i]);
    }

    for (int i = train_size; i < input_data.size(); ++i) {
        test_input_data.push_back(input_data[i]);
        test_target_data.push_back(target_data[i]);
    }

    int numInputs = train_input_data.size();

    vector<double> inputValues, targetValues, resultValues;
    int trainingPass = 0;
    int numEpochs = 1; // Set the number of epochs you want to train for

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        // Loop through your training data and perform forward and backward passes for each data point
        for (int i = 0; i < train_input_data.size(); ++i) {

            const vector<double>& inputValues = train_input_data[i];
            const vector<double>& targetValues = { static_cast<double>(train_target_data[i]) };

            myNetwork.feedForward(inputValues, train_input_data);
            myNetwork.backPropagation(train_input_data, train_target_data);

            std::cout << "Pass: " << i + 1 << "/" << numInputs << std::endl;
        }

        std::cout << "Epoch: " << epoch + 1 << "/" << numEpochs << std::endl;
    }

    // Test the model
    int numCorrect = 0;
    float accuracy = 0.0;
    int truePositives = 0;
    int trueNegatives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < test_input_data.size(); ++i) {
        inputValues.clear();
        for (int j = 0; j < test_input_data[i].size(); ++j) {
            inputValues.push_back(test_input_data[i][j]);
        }
        targetValues.clear();
        targetValues.push_back(test_target_data[i]);

        myNetwork.feedForward(inputValues, test_input_data);

        myNetwork.getResults(resultValues);

        float bias = 0.0;

        if (resultValues[0] - bias >= 0) {
            if (targetValues[0] == 1) {
                numCorrect++;
                truePositives++;
            }
            else {
                trueNegatives++;
            }
        }
        else {
            if (targetValues[0] == 0) {
                numCorrect++;
                falsePositives++;
            }
            else {
                falseNegatives++;
            }
        }
    }

    accuracy = (float)numCorrect / test_input_data.size();

    cout << "Accuracy: " << accuracy * 100 << "%" << endl;
    cout << "True positives: " << truePositives << endl;
    cout << "True negatives: " << trueNegatives << endl;
    cout << "False positives: " << falsePositives << endl;
    cout << "False negatives: " << falseNegatives << endl;

    return 0;
}
