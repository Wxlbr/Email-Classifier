#include <vector>
#include <cmath>
#include <iostream>

#include "model.hpp"
#include "data.hpp"

int main() {
    // Load data from emails.csv
    std::vector<std::vector<double>> input_data;
    std::vector<int> target_data;
    std::vector<std::string> header;
    // ./inc/emailsHotEncoding.csv ./inc/emails.csv
    loadData("./inc/emailsHotEncoding.csv", input_data, target_data, header, "Prediction", 250, 10);

    // Train test split, 80% train, 20% test
    std::vector<std::vector<double>> train_input_data;
    std::vector<int> train_target_data;
    std::vector<std::vector<double>> test_input_data;
    std::vector<int> test_target_data;
    trainTestSplit(input_data, target_data, train_input_data, train_target_data, test_input_data, test_target_data);

    int numInputs = train_input_data[0].size();
    int inputSize = train_input_data.size();

    // numInputs, numHiddenLayers, numNeuronsPerHiddenLayer, numOutputs
    // Keep numHiddenLayers = 1 for now, suspected vanishing/exploding gradients problem
    NeuralNetwork myNetwork(numInputs, 1, 5, 1);

    std::vector<double> inputValues, targetValues, resultValues;
    int trainingPass = 0;
    int numEpochs = 1; // Set the number of epochs you want to train for

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        // Loop through your training data and perform forward and backward passes for each data point
        for (int i = 0; i < train_input_data.size(); ++i) {

            const std::vector<double>& inputValues = train_input_data[i];
            const std::vector<double>& targetValues = { static_cast<double>(train_target_data[i]) };

            myNetwork.feedForward(inputValues, train_input_data);
            myNetwork.backPropagation(train_input_data, train_target_data);

            std::cout << "Pass: " << i + 1 << "/" << inputSize << std::endl;
        }

        std::cout << "Epoch: " << epoch + 1 << "/" << numEpochs << std::endl;
    }

    // Test the model
    int numCorrect = 0;
    int truePositives = 0;
    int trueNegatives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < test_input_data.size(); ++i) {
        inputValues.clear();
        for (int j = 0; j < test_input_data[i].size(); ++j) {
            inputValues.push_back(test_input_data[i][j]);
        }

        int targetValue = test_target_data[i];

        myNetwork.feedForward(inputValues, test_input_data);
        resultValues.push_back(myNetwork.getResults()[0]);
    }

    int threshold = 0;

    // Calculate accuracy
    for (int i = 0; i < resultValues.size(); ++i) {
        if (resultValues[i] >= threshold && test_target_data[i] == 1) {
            numCorrect++;
            truePositives++;
        }
        else if (resultValues[i] < threshold && test_target_data[i] == 0) {
            numCorrect++;
            trueNegatives++;
        }
        else if (resultValues[i] >= threshold && test_target_data[i] == 0) {
            falsePositives++;
        }
        else if (resultValues[i] < threshold && test_target_data[i] == 1) {
            falseNegatives++;
        }
    }

    float accuracy = (float)numCorrect / test_input_data.size();

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    std::cout << "True positives: " << truePositives << std::endl;
    std::cout << "True negatives: " << trueNegatives << std::endl;
    std::cout << "False positives: " << falsePositives << std::endl;
    std::cout << "False negatives: " << falseNegatives << std::endl;

    return 0;
}
