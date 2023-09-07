#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> NeuronLayer;

class Neuron {
private:
    static double learningRate; // [0.0...1.0] overall net training rate
    static double momentum; // [0.0...n] multiplier of last weight change [momentum]
    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    double outputValue;
    vector<Connection> outputWeights;
    unsigned neuronIndex;
    double gradient;

public:
    void setOutputValue(double val) { outputValue = val; }
    double getOutputValue(void) const { return outputValue; }

    void updateInputWeights(NeuronLayer &prevLayer) {
        for(unsigned n = 0; n < prevLayer.size(); ++n)
        {
            Neuron &neuron = prevLayer[n];
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

    double sumDOW(const NeuronLayer &nextLayer) const {
        double sum = 0.0;

        for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
        {
            sum += outputWeights[n].weight * nextLayer[n].gradient;
        }

        return sum;
    }

    void calculateHiddenGradients(const NeuronLayer &nextLayer) {
        double dow = sumDOW(nextLayer);
        gradient = dow * Neuron::transferFunctionDerivative(outputValue);
    }

    void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        gradient = delta * Neuron::transferFunctionDerivative(outputValue);
    }

    double transferFunction(double x) {
        return tanh(x);
    }

    double transferFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    void feedForward(const NeuronLayer &prevLayer) {
        double sum = 0.0;

        for(unsigned n = 0; n < prevLayer.size(); ++n)
        {
            sum += prevLayer[n].getOutputValue() *
                prevLayer[n].outputWeights[neuronIndex].weight;
        }

        outputValue = Neuron::transferFunction(sum);
    }

    Neuron(unsigned numOutputs, unsigned index) {
        for(unsigned c = 0; c < numOutputs; ++c){
            outputWeights.push_back(Connection());
            outputWeights.back().weight = randomWeight();
        }

        neuronIndex = index;
    }
};

class NeuralNetwork {
private:
    vector<NeuronLayer> layers;
    double error;
    double recentAverageError;
    static double recentAverageSmoothingFactor;
public:
    NeuralNetwork(const vector<unsigned> &topology) {
        unsigned numLayers = topology.size();
        for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
            layers.push_back(NeuronLayer());
            unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];

            for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
                layers.back().push_back(Neuron(numOutputs, neuronNum));
                cout << "Made a Neuron!" << endl;
            }

            layers.back().back().setOutputValue(1.0);
        }
    }

    void getResults(vector<double> &resultValues) const {
        resultValues.clear();

        for(unsigned n = 0; n < layers.back().size() - 1; ++n) {
            resultValues.push_back(layers.back()[n].getOutputValue());
        }
    }

    void backPropagation(const std::vector<double> &targetValues) {
        NeuronLayer &outputLayer = layers.back();
        error = 0.0;

        for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
            double delta = targetValues[n] - outputLayer[n].getOutputValue();
            error += delta * delta;
        }

        error /= outputLayer.size() - 1;
        error = sqrt(error);

        recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error)
            / (recentAverageSmoothingFactor + 1.0);

        for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
            outputLayer[n].calculateOutputGradients(targetValues[n]);
        }

        for(unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
            NeuronLayer &hiddenLayer = layers[layerNum];
            NeuronLayer &nextLayer = layers[layerNum + 1];

            for(unsigned n = 0; n < hiddenLayer.size(); ++n) {
                hiddenLayer[n].calculateHiddenGradients(nextLayer);
            }
        }

        for(unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
            NeuronLayer &layer = layers[layerNum];
            NeuronLayer &prevLayer = layers[layerNum - 1];

            for(unsigned n = 0; n < layer.size() - 1; ++n) {
                layer[n].updateInputWeights(prevLayer);
            }
        }
    }

    void feedForward(const vector<double> &inputValues) {
        assert(inputValues.size() == layers[0].size() - 1);

        for(unsigned i = 0; i < inputValues.size(); ++i){
            layers[0][i].setOutputValue(inputValues[i]);
        }

        for(unsigned layerNum = 1; layerNum < layers.size(); ++layerNum){
            NeuronLayer &prevLayer = layers[layerNum - 1];
            for(unsigned n = 0; n < layers[layerNum].size() - 1; ++n){
                layers[layerNum][n].feedForward(prevLayer);
            }
        }
    }

    double getRecentAverageError(void) const { return recentAverageError; }
};

double Neuron::learningRate = 0.1;
double Neuron::momentum = 0.5;
double NeuralNetwork::recentAverageSmoothingFactor = 100.0;

void showVectorValues(string label, vector<double> &values) {
    cout << label << " ";
    for(unsigned i = 0; i < values.size(); ++i) {
        cout << values[i] << " ";
    }
    cout << endl;
}

int main()
{
    // Load data from emails.csv
    std::vector<std::vector<int>> data;

    std::string flagHeader = "Prediction";
    std::vector<std::string> header;
    std::vector<int> labels;

    std::ifstream file("./inc/emailsHotEncoding.csv");

    // Skip the first line (header)
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;

    int flagIndex = -1;
    int i = 0;

    while (std::getline(ss, cell, ',')) {
        if (cell == flagHeader) {
            flagIndex = i;
        } else {
            header.push_back(cell);
        }

        i++;
    }

    if (flagIndex == -1) {
        std::cerr << "Flag header not found: " << flagHeader << std::endl;
        return 1;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        std::vector<int> row;
        int index = 0;

        while (std::getline(ss, cell, ',')) {
            if (index == flagIndex) {
                labels.push_back(std::stoi(cell));
            } else {
                row.push_back(std::stod(cell));
            }
            index++;
        }

        // Shrink row to first x elements
        row.resize(3000);

        data.push_back(row);
    }

    std::vector<std::vector<int>> input_data = data;
    std::vector<int> target_data = labels;

    vector<unsigned> topology = {3000, 5, 1};

    NeuralNetwork myNetwork(topology);

    // Train test split, 80% train, 20% test
    int train_size = (int)(input_data.size() * 0.8);
    int test_size = input_data.size() - train_size;

    std::vector<std::vector<int>> train_input_data;
    std::vector<int> train_target_data;
    std::vector<std::vector<int>> test_input_data;
    std::vector<int> test_target_data;

    for (int i = 0; i < train_size; ++i) {
        train_input_data.push_back(input_data[i]);
        train_target_data.push_back(target_data[i]);
    }

    for (int i = train_size; i < input_data.size(); ++i) {
        test_input_data.push_back(input_data[i]);
        test_target_data.push_back(target_data[i]);
    }

    vector<double> inputValues, targetValues, resultValues;
    int trainingPass = 0;
    int numEpochs = 5; // Set the number of epochs you want to train for

    for (int epoch = 0; epoch < numEpochs; ++epoch) {

        // Loop through your training data and perform forward and backward passes for each data point
        for (int i = 0; i < train_input_data.size(); ++i) {
            inputValues.clear();
            for (int j = 0; j < train_input_data[i].size(); ++j) {
                inputValues.push_back(train_input_data[i][j]);
            }
            targetValues.clear();
            targetValues.push_back(train_target_data[i]);

            myNetwork.feedForward(inputValues);
            myNetwork.backPropagation(targetValues);

            // Report how well the training is working for each data point
            cout << "Epoch " << epoch + 1 << ", Data Point " << i + 1 << ", Recent Average Error: "
                << myNetwork.getRecentAverageError() << endl;
        }
    }

    cout << endl << "Done" << endl;

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
        // showVectorValues(": Inputs :", inputValues);
        myNetwork.feedForward(inputValues);

        // Collect the net's actual results:
        myNetwork.getResults(resultValues);
        showVectorValues("Outputs:", resultValues);

        // Train the net what the outputs should have been:
        myNetwork.backPropagation(targetValues);

        // Report how well the training is working, average over recent
        cout << "Net recent average error: "
             << myNetwork.getRecentAverageError() << endl;

        float bias = 0.0;

        if (resultValues[0] - bias >= 0) {
            if (targetValues[0] == 1) {
                numCorrect++;
                truePositives++;
            } else {
                trueNegatives++;
            }
        } else {
            if (targetValues[0] == 0) {
                numCorrect++;
                falsePositives++;
            } else {
                falseNegatives++;
            }
        }
    }

    accuracy = (float)numCorrect / test_input_data.size();

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "True positives: " << truePositives << std::endl;
    std::cout << "True negatives: " << trueNegatives << std::endl;
    std::cout << "False positives: " << falsePositives << std::endl;
    std::cout << "False negatives: " << falseNegatives << std::endl;

    return 0;
}
