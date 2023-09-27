#include <iostream>
#include <vector>
#include <cmath>

#include "data.hpp"

class LSTMCell {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    // Wi, Wf, Wc, Wo
    std::vector<std::vector<double>> inputWeights;
    std::vector<std::vector<double>> forgetWeights;
    std::vector<std::vector<double>> candidateWeights;
    std::vector<std::vector<double>> outputWeights;

    // Ui, Uf, Uc, Uo
    std::vector<std::vector<double>> inputRecurrentWeights;
    std::vector<std::vector<double>> forgetRecurrentWeights;
    std::vector<std::vector<double>> candidateRecurrentWeights;
    std::vector<std::vector<double>> outputRecurrentWeights;

    // bi, bf, bc, bo
    std::vector<double> inputBiases;
    std::vector<double> forgetBiases;
    std::vector<double> candidateBiases;
    std::vector<double> outputBiases;

    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<double> states;


public:
    LSTMCell(int inputSize, int hiddenSize, int outputSize) : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        // Initialize weights
        inputWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        forgetWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        candidateWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        outputWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));

        // Initialize recurrent weights
        inputRecurrentWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        forgetRecurrentWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        candidateRecurrentWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));
        outputRecurrentWeights = std::vector<std::vector<double>>(4, std::vector<double>(hiddenSize, 0));

        // Initialize biases
        inputBiases = std::vector<double>(hiddenSize, 0);
        forgetBiases = std::vector<double>(hiddenSize, 0);
        candidateBiases = std::vector<double>(hiddenSize, 0);
        outputBiases = std::vector<double>(hiddenSize, 0);

        // Initialize inputs, outputs, and states
        inputs = std::vector<double>(inputSize, 0);
        outputs = std::vector<double>(outputSize, 0);
        states = std::vector<double>(hiddenSize, 0);
    }

    void setInitialStates(std::vector<double> states) {
        // Populate with zeros of size hiddenSize
        this->states = std::vector<double>(hiddenSize, 0);
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double tanh(double x) {
        return tanh(x);
    }

    void forward(std::vector<double> inputs) {
        this->inputs = inputs;

        // Forget gate
        std::vector<double> forgetGate(hiddenSize, 0);
        for (int i = 0; i < hiddenSize; i++) {
            forgetGate[i] = sigmoid(inputRecurrentWeights[1][i] * states[i] + forgetRecurrentWeights[1][i] * inputs[i] + forgetBiases[i]);
        }

        // Input gate
        std::vector<double> inputGate(hiddenSize, 0);
        for (int i = 0; i < hiddenSize; i++) {
            inputGate[i] = sigmoid(inputRecurrentWeights[0][i] * states[i] + inputRecurrentWeights[0][i] * inputs[i] + inputBiases[i]);
        }

        // Candidate state
        std::vector<double> candidateState(hiddenSize, 0);
        for (int i = 0; i < hiddenSize; i++) {
            candidateState[i] = tanh(inputRecurrentWeights[2][i] * states[i] + candidateRecurrentWeights[2][i] * inputs[i] + candidateBiases[i]);
        }

        // Update state
        for (int i = 0; i < hiddenSize; i++) {
            states[i] = forgetGate[i] * states[i] + inputGate[i] * candidateState[i];
        }

        // Output gate
        std::vector<double> outputGate(hiddenSize, 0);
        for (int i = 0; i < hiddenSize; i++) {
            outputGate[i] = sigmoid(inputRecurrentWeights[3][i] * states[i] + outputRecurrentWeights[3][i] * inputs[i] + outputBiases[i]);
        }

        // Update output
        for (int i = 0; i < hiddenSize; i++) {
            outputs[i] = outputGate[i] * tanh(states[i]);
        }
    }

    std::vector<double> getOutputs() {
        return outputs;
    }

    std::vector<double> getStates() {
        return states;
    }
};

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;

public:
    LSTMNetwork(int inputSize, int hiddenSize, int outputSize) {
        // Initialize LSTM cells
        LSTMCell cell(inputSize, hiddenSize, outputSize);
        cells.push_back(cell);
    }

    void forward(std::vector<double> inputs) {
        std::cout << "Forward pass" << std::endl;
        // Forward pass through each cell
        std::vector<double> currentInput = inputs;

        // Iterate through the LSTM cells and perform forward pass
        for (auto& cell : cells) {
            cell.forward(currentInput);
            currentInput = cell.getOutputs(); // Update input for the next cell
        }
    }

    void eval(std::vector<double> inputs, int target) {
        std::cout << "Evaluating" << std::endl;

        forward(inputs);

        std::cout << "Forward pass complete" << std::endl;

        std::cout << cells.size() << std::endl;

        LSTMCell lastCell = cells[cells.size() - 1];

        std::cout << "Got last cell" << std::endl;

        // Get outputs from last cell
        std::vector<double> outputs = lastCell.getOutputs();

        std::cout << "Got outputs" << std::endl;

        // Get max value from outputs
        double max = outputs[0];
        int maxIndex = 0;
        for (int i = 1; i < outputs.size(); i++) {
            if (outputs[i] > max) {
                max = outputs[i];
                maxIndex = i;
            }
        }

        // Print prediction
        std::cout << "Prediction: " << maxIndex << " Target: " << target << std::endl;
    }


};

int main() {
    // Load data from emails.csv
    std::vector<std::vector<double>> input_data;
    std::vector<int> target_data;
    std::vector<std::string> header;
    // ./inc/emailsHotEncoding.csv ./inc/emails.csv
    loadData("./inc/emailsHotEncoding.csv", input_data, target_data, header, "Prediction", 250, 10); // 250, 10

    // Train test split, 80% train, 20% test
    std::vector<std::vector<double>> train_input_data;
    std::vector<int> train_target_data;
    std::vector<std::vector<double>> test_input_data;
    std::vector<int> test_target_data;
    trainTestSplit(input_data, target_data, train_input_data, train_target_data, test_input_data, test_target_data);

    int inputSize = train_input_data[0].size();
    int hiddenSize = inputSize;
    int outputSize = 1;

    // Initialize LSTM network
    LSTMNetwork network(inputSize, hiddenSize, outputSize);

    // Train LSTM network
    for (int i = 0; i < train_input_data.size(); i++) {
        network.forward(train_input_data[i]);
    }

    std::cout << "Training complete" << std::endl;

    for (int i = 0; i < test_input_data.size(); i++) {
        network.eval(test_input_data[i], test_target_data[i]);
    }

    std::cout << "Testing complete" << std::endl;

    return 0;
}