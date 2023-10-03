#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "data.hpp" // Assuming this contains your data loading functions

class LSTMCell {
private:
    int inputSize;
    int hiddenSize;

    // Initialize
    std::vector<double> x; // Input values
    std::vector<double> h; // Hidden state
    std::vector<double> c; // Cell state

    std::vector<double> o; // Output gate
    std::vector<double> wo; // Output gate weights
    std::vector<double> bo; // Output gate bias

    std::vector<double> i; // Input gate
    std::vector<double> wi; // Input gate weights
    std::vector<double> bi; // Input gate bias

    std::vector<double> f; // Forget gate
    std::vector<double> wf; // Forget gate weights
    std::vector<double> bf; // Forget gate bias

    std::vector<double> cHat; // Candidate cell state
    std::vector<double> wc; // Candidate cell state weights
    std::vector<double> bc; // Candidate cell state bias

public:
    LSTMCell(int inputSize, int hiddenSize) : inputSize(inputSize), hiddenSize(hiddenSize) {
        // Initialize LSTM cell
        initializeWeightsAndBiases();

        // Initialize input values, hidden state, and cell state
        x.resize(inputSize);
        h.resize(hiddenSize);
        c.resize(hiddenSize);
    }

    void initializeWeightsAndBiases() {
        // Initialize weights and biases using Xavier initialization

        double mean = 0;
        double variance = 2.0 / (inputSize + hiddenSize);
        double standardDeviation = sqrt(variance);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(mean, standardDeviation);

        wo.resize(hiddenSize);
        bo.resize(1);
        wi.resize(hiddenSize);
        bi.resize(1);
        wf.resize(hiddenSize);
        bf.resize(1);
        wc.resize(hiddenSize);
        bc.resize(1);

        for (int i = 0; i < hiddenSize; i++) {
            wo[i] = dist(gen);
            wi[i] = dist(gen);
            wf[i] = dist(gen);
            wc[i] = dist(gen);
        }

        bo[0] = dist(gen);
        bi[0] = dist(gen);
        bf[0] = dist(gen);
        bc[0] = dist(gen);
    }

    double tanh(double x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    void forgetGate() {
        // ft = sigmoid(Wf * [ht-1, xt] + bf)

        int totalSize = hiddenSize + inputSize;
        f.resize(totalSize);

        for (int i = 0; i < totalSize; i++) {
            f[i] = sigmoid(wf[i] * h[i] + wf[i + totalSize] * x[i] + bf[0]);
        }
    }

    void inputGate() {
        // it = sigmoid(Wi * [ht-1, xt] + bi)

        int totalSize = hiddenSize + inputSize;
        i.resize(totalSize);

        for (int j = 0; j < totalSize; j++) {
            i[j] = sigmoid(wi[j] * h[j] + wi[j + totalSize] * x[j] + bi[0]);
        }
    }

    void candidateCellState() {
        // cHat = tanh(Wc * [ht-1, xt] + bc)

        int totalSize = hiddenSize + inputSize;
        cHat.resize(totalSize);

        for (int i = 0; i < totalSize; i++) {
            cHat[i] = tanh(wc[i] * h[i] + wc[i + totalSize] * x[i] + bc[0]);
        }
    }

    void updateCellState() {
        // ct = ft * ct-1 + it * cHat

        int totalSize = hiddenSize + inputSize;
        c.resize(totalSize);

        for (int j = 0; j < totalSize; j++) {
            c[j] = f[j] * c[j] + i[j] * cHat[j];
        }
    }

    void outputGate() {
        // ot = sigmoid(Wo * [ht-1, xt] + bo)

        int totalSize = hiddenSize + inputSize;
        o.resize(totalSize);

        for (int i = 0; i < totalSize; i++) {
            o[i] = sigmoid(wo[i] * h[i] + wo[i + totalSize] * x[i] + bo[0]);
        }
    }

    void updateHiddenState() {
        // ht = ot * tanh(ct)

        int totalSize = hiddenSize + inputSize;
        h.resize(totalSize);

        for (int i = 0; i < totalSize; i++) {
            h[i] = o[i] * tanh(c[i]);
        }
    }

    void forward(const std::vector<double>& inputs) {
        // Forward pass through LSTM cell
        x = inputs;

        // Forget gate
        forgetGate();

        // Input gate
        inputGate();

        // Candidate cell state
        candidateCellState();

        // Cell state
        updateCellState();

        // Output gate
        outputGate();

        // Hidden state
        updateHiddenState();
    }

    void backward(const std::vector<double>& dNextHiddenState, const std::vector<double>& dNextCellState) {
        // Backward pass through LSTM cell

        // Compute gradient loss

        // Compute gradients

        // Accumulate gradients

        // Compute gradient loss with respect to input and hidden state

        // Update weights and biases with gradient descent
        //      using learning rate as a multiplier
    }

    // Get output
    std::vector<double> getOutput() {
        return h;
    }

    std::vector<double> getSigmoidOutput() {
        std::vector<double> sigmoidOutput;
        for (int i = 0; i < h.size(); i++) {
            sigmoidOutput.push_back(sigmoid(h[i]));
        }
        return sigmoidOutput;
    }
};

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;

public:
    LSTMNetwork(int inputSize, int hiddenSize, int outputSize) {
        // Initialize LSTM cells
        LSTMCell cell(inputSize, hiddenSize);
        cells.push_back(cell);
    }

    void forward(std::vector<double>& inputs) {
        // Forward pass through LSTM cells
        for (int i = 0; i < cells.size(); i++) {
            cells[i].forward(inputs);
        }
    }

    void BPTT(std::vector<std::vector<double>>& inputs, std::vector<int> targets) {
        // Backward pass through time

        // TEMP
        // Loop over inputs
        std::vector<double> input = inputs[0];

        // Forward pass
        forward(inputs);

        // Compute loss
        // Mean squared or entropy

        // Backward pass
        //    initialise initial gradients and cell states
        //    for each LSTM cell to 0 in a vector
        //    loop in reverse order through LSTM cells

        for (int i = cells.size() - 1; i >= 0; i--) {
            // Pass gradients and cell states to previous LSTM cell
            LSTMCell cell = cells[i];
            cell.backward( /* gradients and cell states */ );
        }
    }

    double loss(int target, int prediction) {
        // Calculate loss
        double loss = 0.5 * pow(target - prediction, 2);
        return loss;
    }

    void eval(const std::vector<std::vector<double>>& inputs, std::vector<int> targets) {
        // Evaluate LSTM network
        std::vector<double> predictions;
        for (int i = 0; i < inputs.size(); i++) {
            forward(inputs[i]);
            // std::vector<double> output = cells[0].getOutput();
            std::vector<double> output = cells.back().getOutput();
            predictions.push_back(output[0]);
        }

        double totalLoss = 0;
        for (int i = 0; i < predictions.size(); i++) {
            double lossValue = loss(targets[i], predictions[i]);
            totalLoss += lossValue;
        }

        double averageLoss = totalLoss / predictions.size();

        std::cout << "Average loss: " << averageLoss << std::endl;

        int correctPositices = 0;
        int correctNegatives = 0;
        int incorrectPositives = 0;
        int incorrectNegatives = 0;

        for (int i = 0; i < predictions.size(); i++) {
            std::cout << predictions[i] << " ";
            if (predictions[i] > 0.5) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }

            if (predictions[i] == 1 && targets[i] == 1) {
                correctPositices++;
            } else if (predictions[i] == 0 && targets[i] == 0) {
                correctNegatives++;
            } else if (predictions[i] == 1 && targets[i] == 0) {
                incorrectPositives++;
            } else if (predictions[i] == 0 && targets[i] == 1) {
                incorrectNegatives++;
            }
        }

        double accuracy = (correctPositices + correctNegatives) * 100 / (double) predictions.size();

        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Correct positives: " << correctPositices << std::endl;
        std::cout << "Correct negatives: " << correctNegatives << std::endl;
        std::cout << "Incorrect positives: " << incorrectPositives << std::endl;
        std::cout << "Incorrect negatives: " << incorrectNegatives << std::endl;
    }
};

int main() {
    // Load data from emails.csv
    std::vector<std::vector<double>> input_data;
    std::vector<int> target_data;
    std::vector<std::string> header;
    loadData("./inc/emailsHotEncoding.csv", input_data, target_data, header, "Prediction", 250, 10);

    // Train test split, 80% train, 20% test
    std::vector<std::vector<double>> train_input_data;
    std::vector<int> train_target_data;
    std::vector<std::vector<double>> test_input_data;
    std::vector<int> test_target_data;
    trainTestSplit(input_data, target_data, train_input_data, train_target_data, test_input_data, test_target_data);

    int inputSize = train_input_data[0].size();
    int hiddenSize = inputSize;
    int outputSize = hiddenSize;

    // Initialize LSTM network
    LSTMNetwork network(inputSize, hiddenSize, outputSize);

    // Train LSTM network
    for (int i = 0; i < train_input_data.size(); i++) {
        network.forward(train_input_data[i]);
        std::cout << "Forward pass " << i << "/" << train_input_data.size() << " complete" << std::endl;
    }

    std::cout << "Training complete" << std::endl;

    network.eval(test_input_data, test_target_data);

    std::cout << "Testing complete" << std::endl;

    return 0;
}
