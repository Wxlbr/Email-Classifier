#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "data.hpp"
#include "model.hpp"
#include "model2.hpp"
#include "tcn.hpp"

int main() {
    std::cout << "Starting program..." << std::endl;

    // Load data from emails.csv
    std::vector<std::vector<int>> data;

    std::string flagHeader = "Prediction";
    std::vector<std::string> header;
    std::vector<int> labels;

    std::ifstream file("./inc/emails.csv");

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

        data.push_back(row);
    }

    std::vector<std::vector<int>> input_data = data;
    std::vector<int> target_data = labels;

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

    std::cout << "Data size: " << train_input_data.size() << std::endl;

    std::cout << "Input size: " << train_input_data[0].size() << std::endl;

    // Define RNN parameters
    int inputSize = 1; // Number of input features
    int hiddenLayerSize = 10;
    int outputLayerSize = 1;
    int numEpochs = 10;
    float learningRate = 0.01;
    int sequenceLength = 5; // Number of time steps


    // Create RNN model
    // RecurrentNeuralNetwork model(inputSize, hiddenLayerSize, outputLayerSize, numEpochs, learningRate, sequenceLength);
    RandomForest model(100, 200, 200);
    // TCNModel model(50, std::vector<int>{5, 5, 5, 5, 5});

    std::cout << "Training model" << std::endl;

    // Train the model
    model.train(train_input_data, train_target_data);

    std::cout << "Model trained" << std::endl;

    model.saveModel("./inc/forest100_200_200.bin");

    std::cout << "Data size: " << test_input_data.size() << std::endl;

    // Test the model

    int true_positives = 0;
    int false_positives = 0;
    int true_negatives = 0;
    int false_negatives = 0;

    int correct = 0;
    float accuracy = 0.0;

    for (size_t i = 0; i < test_input_data.size(); ++i) {
        std::vector<int> input = test_input_data[i];
        int target = test_target_data[i];

        // std::vector<int> predictions = model.predict(input);
        // for (int i = 0; i < predictions.size(); ++i) {
        //     std::cout << predictions[i] << " ";
        // }
        // int prediction = (predictions[0] >= 0.5) ? 1 : 0;
        // double prediction = model.predict(input);
        int prediction = model.predict(input);
        // std::cout << prediction << " ";
        // prediction = (prediction >= 0.5) ? 1 : 0;

        if (prediction == target) {
            correct++;
            if (prediction == 1) {
                true_positives++;
            } else {
                true_negatives++;
            }
        } else {
            if (prediction == 1) {
                false_positives++;
            } else {
                false_negatives++;
            }
        }
    }
    std::cout << std::endl;

    accuracy = (float)correct / (float)test_input_data.size();

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "True positives: " << true_positives << std::endl;
    std::cout << "False positives: " << false_positives << std::endl;
    std::cout << "True negatives: " << true_negatives << std::endl;
    std::cout << "False negatives: " << false_negatives << std::endl;

    return 0;
}
