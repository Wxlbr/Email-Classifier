#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

#include "model.hpp"

using namespace std;

int main() {
    // Load data from emails.csv
    vector<vector<double>> data;
    string flagHeader = "Prediction";
    vector<string> header;
    vector<int> labels;
    ifstream file("./inc/emails.csv"); // ./inc/emailsHotEncoding.csv
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
    // Keep numHiddenLayers = 1 for now, suspected vanishing/exploding gradient problem
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

        // std::cout << "ResultValues: ";
        // for (int i = 0; i < resultValues.size(); ++i) {
        //     std::cout << resultValues[i] << " ";
        // }

        if (resultValues[0] >= 0) {
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
