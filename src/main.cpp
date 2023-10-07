#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <algorithm> // Added for std::min

#include "data.hpp"

class LSTMCell {
private:
    int inputSize;
    int hiddenSize;
    double learningRate = 0.01;
    double clipValue = 1.0;

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

    // Backpropagation through time
    std::vector<double> dOutputGate; // Output gate derivative
    std::vector<double> dWo; // Output gate weights derivative
    std::vector<double> dBo; // Output gate bias derivative

    std::vector<double> dInputGate; // Input gate derivative
    std::vector<double> dWi; // Input gate weights derivative
    std::vector<double> dBi; // Input gate bias derivative

    std::vector<double> dForgetGate; // Forget gate derivative
    std::vector<double> dWf; // Forget gate weights derivative
    std::vector<double> dBf; // Forget gate bias derivative

    std::vector<double> dcHat; // Cell state derivative
    std::vector<double> dc; // Cell state derivative
    std::vector<double> dWc; // Cell state weights derivative
    std::vector<double> dBc; // Cell state bias derivative

    std::vector<double> dHiddenState; // Hidden state derivative

    std::vector<double> daPrev; // Previous hidden state derivative
    std::vector<double> dcPrev; // Previous cell state derivative
    std::vector<double> dxt; // Input derivative

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
        double variance = 2.0 / inputSize; // (inputSize + hiddenSize);
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

        for (int i = 0; i < hiddenSize; i++) {
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

        for (int i = 0; i < hiddenSize; i++) {
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

        for (int i = 0; i < hiddenSize; i++) {
            o[i] = sigmoid(wo[i] * h[i] + wo[i + totalSize] * x[i] + bo[0]);
        }
    }

    void updateHiddenState() {
        // ht = ot * tanh(ct)

        int totalSize = hiddenSize + inputSize;
        h.resize(totalSize);

        for (int i = 0; i < hiddenSize; i++) {
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

    void derivativeOutputGate(std::vector<double>& dNextHiddenState) {
        // Derivative of output gate
        // std::cout << "Derivative of output gate" << std::endl;

        dOutputGate.resize(hiddenSize);

        // do = dh * tanh(ct) * o * (1 - o)
        for (int i = 0; i < hiddenSize; i++) {
            dOutputGate[i] = dNextHiddenState[i] * tanh(c[i]) * o[i] * (1.0 - o[i]);
        }
    }

    void derivativeInputGate() {
        // Derivative of input gate
        // std::cout << "Derivative of input gate" << std::endl;

        dInputGate.resize(hiddenSize);

        // di = dc * cHat * i * (1 - i)
        for (int i = 0; i < hiddenSize; i++) {
            dInputGate[i] = dc[i] * cHat[i] * x[i] * (1.0 - x[i]);
        }
    }

    void derivativeForgetGate() {
        // Derivative of forget gate

        dForgetGate.resize(hiddenSize);

        // df = dc * ct-1 * f * (1 - f)
        for (int i = 0; i < hiddenSize; i++) {
            dForgetGate[i] = dc[i] * c[i] * f[i] * (1.0 - f[i]);
        }
    }

    void derivativeCandidateCellState() {
        // Derivative of candidate cell state

        dcHat.resize(hiddenSize);

        // dcHat = dc * i * (1 - cHat^2)
        for (int i = 0; i < hiddenSize; i++) {
            dcHat[i] = dc[i] * x[i] * (1.0 - cHat[i] * cHat[i]);
        }
    }

    void backward(std::vector<double>& dNextHiddenState, std::vector<double>& dNextCellState) {
        // Backward pass through LSTM cell
        // std::cout << "Backward pass through LSTM cell" << std::endl;

        // Caluclate dc
        dc.resize(hiddenSize);

        for (int i = 0; i < hiddenSize; i++) {
            dc[i] = dNextHiddenState[i] * o[i] * (1.0 - tanh(tanh(c[i])) * tanh(tanh(c[i]))) + dNextCellState[i];
        }

        // Calculate Gate Derivatives
        derivativeOutputGate(dNextHiddenState);
        // std::cout << "Derivative of output gate complete" << std::endl;
        derivativeCandidateCellState(); // Check order
        // std::cout << "Derivative of candidate cell state complete" << std::endl;
        derivativeInputGate();
        // std::cout << "Derivative of input gate complete" << std::endl;
        derivativeForgetGate();
        // std::cout << "Derivative of forget gate complete" << std::endl;

        // Calculate Weight Derivatives

        dWo.resize(hiddenSize);
        dWi.resize(hiddenSize);
        dWf.resize(hiddenSize);
        dWc.resize(hiddenSize);

        for (int i = 0; i < hiddenSize; i++) {
            dWo[i] = dOutputGate[i] * h[i];
            dWi[i] = dInputGate[i] * h[i];
            dWf[i] = dForgetGate[i] * h[i];
            dWc[i] = dcHat[i] * h[i];
        }

        // std::cout << "Derivative of weights complete" << std::endl;

        // Calculate Bias Derivatives
        dBo.resize(1, 0.0);
        dBi.resize(1, 0.0);
        dBf.resize(1, 0.0);
        dBc.resize(1, 0.0);

        for (int i = 0; i < hiddenSize; i++) {
            dBo[0] += dOutputGate[i];
            dBi[0] += dInputGate[i];
            dBf[0] += dForgetGate[i];
            dBc[0] += dcHat[i];
        }

        // std::cout << "Derivative of biases complete" << std::endl;

        // Calculate Hidden State Derivatives (da_prev, dc_prev, dxt)
        daPrev.resize(hiddenSize * 2, 0.0);
        dcPrev.resize(hiddenSize, 0.0);
        dxt.resize(hiddenSize * 2, 0.0);

        for (int i = 0; i < hiddenSize; i++) {
            daPrev[i] = wf[i] * dForgetGate[i] + wi[i] * dInputGate[i] + wc[i] * dcHat[i] + wo[i] * dOutputGate[i];
            daPrev[i + hiddenSize] = wf[i + hiddenSize] * dForgetGate[i] + wi[i + hiddenSize] * dInputGate[i] + wc[i + hiddenSize] * dcHat[i] + wo[i + hiddenSize] * dOutputGate[i];
            dcPrev[i] = dNextCellState[i] * f[i] + o[i] * (1.0 - tanh(c[i]) * tanh(c[i])) * f[i] * dNextHiddenState[i];
            dxt[i] = wf[i] * dForgetGate[i] + wi[i] * dInputGate[i] + wc[i] * dcHat[i] + wo[i] * dOutputGate[i];
            dxt[i + hiddenSize] = wf[i + hiddenSize] * dForgetGate[i] + wi[i + hiddenSize] * dInputGate[i] + wc[i + hiddenSize] * dcHat[i] + wo[i + hiddenSize] * dOutputGate[i];
        }

        // std::cout << "Derivative of hidden state complete" << std::endl;

        // Clip gradients to prevent exploding gradients
        for (int i = 0; i < hiddenSize; i++) {
            dWo[i] = std::min(dWo[i], clipValue);
            dWo[i] = std::max(dWo[i], -clipValue);
            dWi[i] = std::min(dWi[i], clipValue);
            dWi[i] = std::max(dWi[i], -clipValue);
            dWf[i] = std::min(dWf[i], clipValue);
            dWf[i] = std::max(dWf[i], -clipValue);
            dWc[i] = std::min(dWc[i], clipValue);
            dWc[i] = std::max(dWc[i], -clipValue);
        }

        dBo[0] = std::min(dBo[0], clipValue);
        dBo[0] = std::max(dBo[0], -clipValue);
        dBi[0] = std::min(dBi[0], clipValue);
        dBi[0] = std::max(dBi[0], -clipValue);
        dBf[0] = std::min(dBf[0], clipValue);
        dBf[0] = std::max(dBf[0], -clipValue);
        dBc[0] = std::min(dBc[0], clipValue);
        dBc[0] = std::max(dBc[0], -clipValue);

        updateWeightsAndBiases();

        // std::cout << "Update weights and biases complete" << std::endl;

        std::cout << "Output Gradient" << std::endl;
        for (int i = 0; i < dOutputGate.size(); i++) {
            std::cout << dOutputGate[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Input Gradient" << std::endl;
        for (int i = 0; i < dInputGate.size(); i++) {
            std::cout << dInputGate[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Forget Gradient" << std::endl;
        for (int i = 0; i < dForgetGate.size(); i++) {
            std::cout << dForgetGate[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Cell State Gradient" << std::endl;
        for (int i = 0; i < dcHat.size(); i++) {
            std::cout << dcHat[i] << " ";
        }
        std::cout << std::endl;

        // std::cout << "Backward pass through LSTM cell complete" << std::endl;
    }

    // Get output
    std::vector<double> getOutput() {
        return h;
    }

    // Get o
    std::vector<double> getOutputGate() {
        return o;
    }

    // Get c
    std::vector<double> getCellState() {
        return c;
    }

    std::vector<double> getSigmoidOutput() {
        std::vector<double> sigmoidOutput;
        for (int i = 0; i < h.size(); i++) {
            sigmoidOutput.push_back(sigmoid(h[i]));
        }
        return sigmoidOutput;
    }

    void updateWeightsAndBiases() {
        // Clip gradients to a maximum value
        const double clip_value = 1.0; // You can adjust this value

        for (int i = 0; i < hiddenSize; i++) {
            // Check if gradients are very small
            if (std::abs(dWo[i]) < 1e-6) {
                dWo[i] = 0.0; // Set small gradients to zero
            } else {
                // Clip gradient values
                dWo[i] = std::min(dWo[i], clip_value);
                dWo[i] = std::max(dWo[i], -clip_value);
            }

            if (std::abs(dWi[i]) < 1e-6) {
                dWi[i] = 0.0;
            } else {
                dWi[i] = std::min(dWi[i], clip_value);
                dWi[i] = std::max(dWi[i], -clip_value);
            }

            if (std::abs(dWf[i]) < 1e-6) {
                dWf[i] = 0.0;
            } else {
                dWf[i] = std::min(dWf[i], clip_value);
                dWf[i] = std::max(dWf[i], -clip_value);
            }

            if (std::abs(dWc[i]) < 1e-6) {
                dWc[i] = 0.0;
            } else {
                dWc[i] = std::min(dWc[i], clip_value);
                dWc[i] = std::max(dWc[i], -clip_value);
            }

            // Update weights
            wo[i] -= learningRate * dWo[i];
            wi[i] -= learningRate * dWi[i];
            wf[i] -= learningRate * dWf[i];
            wc[i] -= learningRate * dWc[i];
        }

        // Clip bias gradients and update biases
        dBo[0] = std::min(dBo[0], clip_value);
        dBo[0] = std::max(dBo[0], -clip_value);
        dBi[0] = std::min(dBi[0], clip_value);
        dBi[0] = std::max(dBi[0], -clip_value);
        dBf[0] = std::min(dBf[0], clip_value);
        dBf[0] = std::max(dBf[0], -clip_value);
        dBc[0] = std::min(dBc[0], clip_value);
        dBc[0] = std::max(dBc[0], -clip_value);

        // Update biases
        bo[0] -= learningRate * dBo[0];
        bi[0] -= learningRate * dBi[0];
        bf[0] -= learningRate * dBf[0];
        bc[0] -= learningRate * dBc[0];
    }


    std::vector<double> previousDerivativeHiddenState() {
        return daPrev;
    }

    std::vector<double> previousDerivativeCellState() {
        return dcPrev;
    }
};

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;

    std::vector<double> dNextHiddenState;
    std::vector<double> dNextCellState;

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

    void BPTT(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets, double learning_rate) {
        // Backward pass through time
        std::cout << "Backward pass through time" << std::endl;

        dNextHiddenState.resize(cells[0].getOutput().size(), 0.0);
        dNextCellState.resize(cells[0].getOutput().size(), 0.0);

        for (int i = 0; i < cells[0].getOutput().size(); i++) {
            dNextHiddenState[i] = (cells[0].getOutput()[i] - targets[i]);
            dNextCellState[i] = (dNextHiddenState[i] * cells[0].getOutputGate()[i] * (1.0 - tanh(cells[0].getCellState()[i]) * tanh(cells[0].getCellState()[i])));
        }

        for (int i = cells.size() - 1; i >= 0; i--) {
            // Pass gradients and cell states to the previous LSTM cell
            LSTMCell& cell = cells[i];
            cell.backward(dNextHiddenState, dNextCellState);

            std::cout << "Backward pass " << i << "/" << cells.size() << " complete" << std::endl;

            // Calculate gradients for the previous time step
            dNextHiddenState = cell.previousDerivativeCellState();
            dNextCellState = cell.previousDerivativeHiddenState();

            std::cout << "Backward pass " << i << "/" << cells.size() << " complete" << std::endl;
        }
    }

    double loss(int target, double prediction) {
        // Calculate loss
        double loss = 0.5 * pow(target - prediction, 2);
        return loss;
    }

    void eval(std::vector<std::vector<double>>& inputs, std::vector<int>& targets) {
        // Evaluate LSTM network
        std::vector<double> predictions;
        for (int i = 0; i < inputs.size(); i++) {
            forward(inputs[i]);
            std::vector<double> output = cells[0].getOutput();
            predictions.push_back(output[0]);
        }

        double totalLoss = 0;
        int correctPositives = 0;
        int correctNegatives = 0;
        int incorrectPositives = 0;
        int incorrectNegatives = 0;

        std::vector<double> uniquePredictions;

        for (int i = 0; i < predictions.size(); i++) {
            double lossValue = loss(targets[i], predictions[i]);
            totalLoss += lossValue;

            if (std::find(uniquePredictions.begin(), uniquePredictions.end(), predictions[i]) == uniquePredictions.end()) {
                uniquePredictions.push_back(predictions[i]);
            }

            if (predictions[i] > 0.5) {
                predictions[i] = 1;
            } else {
                predictions[i] = 0;
            }

            if (predictions[i] == 1 && targets[i] == 1) {
                correctPositives++;
            } else if (predictions[i] == 0 && targets[i] == 0) {
                correctNegatives++;
            } else if (predictions[i] == 1 && targets[i] == 0) {
                incorrectPositives++;
            } else if (predictions[i] == 0 && targets[i] == 1) {
                incorrectNegatives++;
            }
        }

        double averageLoss = totalLoss / predictions.size();
        double accuracy = (correctPositives + correctNegatives) * 100 / (double) predictions.size();

        std::cout << "Average loss: " << averageLoss << std::endl;
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Correct positives: " << correctPositives << std::endl;
        std::cout << "Correct negatives: " << correctNegatives << std::endl;
        std::cout << "Incorrect positives: " << incorrectPositives << std::endl;
        std::cout << "Incorrect negatives: " << incorrectNegatives << std::endl;

        std::cout << "Unique predictions: ";
        for (int i = 0; i < uniquePredictions.size(); i++) {
            std::cout << uniquePredictions[i] << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    // Load data from emails.csv
    std::vector<std::vector<double>> input_data;
    std::vector<int> target_data;
    std::vector<std::string> header;
    loadData("./inc/emails.csv", input_data, target_data, header, "Prediction", 250, 10);

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
    double learning_rate = 0.01; // You can adjust the learning rate
    for (int i = 0; i < train_input_data.size(); i++) {
        network.forward(train_input_data[i]);
        network.BPTT(train_input_data, train_target_data, learning_rate);
        std::cout << "Forward pass " << i << "/" << train_input_data.size() << " complete" << std::endl;
    }

    std::cout << "Training complete" << std::endl;

    network.eval(test_input_data, test_target_data);

    std::cout << "Testing complete" << std::endl;

    return 0;
}
