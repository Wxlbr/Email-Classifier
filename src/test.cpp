#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "data.hpp" // Assuming this contains your data loading functions

class LSTMCell {
private:
    int inputSize;
    int hiddenSize;
    double learningRate = 0.1; // TODO: Make this a parameter

    // TODO: Add momentum

    // initialise
    std::vector<double> x; // Input values
    std::vector<double> h; // Hidden state
    std::vector<double> c; // Cell state

    std::vector<double> o; // Output gate
    std::vector<double> wo; // Output gate weights
    std::vector<double> bo; // Output gate bias
    std::vector<double> dWo; // Gradient of output gate weights
    std::vector<double> dBo; // Gradient of output gate bias

    std::vector<double> i; // Input gate
    std::vector<double> wi; // Input gate weights
    std::vector<double> bi; // Input gate bias
    std::vector<double> dWi; // Gradient of input gate weights
    std::vector<double> dBi; // Gradient of input gate bias

    std::vector<double> f; // Forget gate
    std::vector<double> wf; // Forget gate weights
    std::vector<double> bf; // Forget gate bias
    std::vector<double> dWf; // Gradient of forget gate weights
    std::vector<double> dBf; // Gradient of forget gate bias

    std::vector<double> cHat; // Candidate cell state
    std::vector<double> wc; // Candidate cell state weights
    std::vector<double> bc; // Candidate cell state bias
    std::vector<double> dWc; // Gradient of candidate cell state weights
    std::vector<double> dBc; // Gradient of candidate cell state bias

public:
    LSTMCell(int inputSize, int hiddenSize) : inputSize(inputSize), hiddenSize(hiddenSize) {
        // initialise LSTM cell
        x.resize(inputSize);
        h.resize(hiddenSize);
        c.resize(hiddenSize);  // TODO: Initialise cell states

        initialiseWeightsAndBiases();
    }

    void initialiseWeightsAndBiases() {
        // initialise weights and biases using Xavier initialization

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

    void updateCellState(std::vector<double> prevCellState) {
        c = prevCellState;
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

    void initialiseGradients() {
        int totalSize = hiddenSize + inputSize;

        dWo.resize(totalSize);
        dBo.resize(1);
        dWi.resize(totalSize);
        dBi.resize(1);
        dWf.resize(totalSize);
        dBf.resize(1);
        dWc.resize(totalSize);
        dBc.resize(1);

        for (int i = 0; i < totalSize; i++) {
            dWo[i] = 0.0;
            dWi[i] = 0.0;
            dWf[i] = 0.0;
            dWc[i] = 0.0;
        }

        dBo[0] = 0.0;
        dBi[0] = 0.0;
        dBf[0] = 0.0;
        dBc[0] = 0.0;
    }

    void backward(std::vector<double>& dHNext, std::vector<double>& dCNext) {
        int totalSize = hiddenSize + inputSize;

        for (int t = totalSize - 1; t >= 0; t--) {
            double dO = dHNext[t] * tanh(c[t]);
            double dOInput = dO * o[t] * (1.0 - o[t]);
            dWo[t] += dOInput * x[t];
            dBo[0] += dOInput;

            double dC = dHNext[t] * o[t] * (1.0 - tanh(c[t]) * tanh(c[t])) + dCNext[t];

            double dI = dC * cHat[t] * i[t] * (1.0 - i[t]);
            dWi[t] += dI * x[t];
            dBi[0] += dI;

            double dF = dC * c[t - 1] * f[t] * (1.0 - f[t]);
            dWf[t] += dF * x[t];
            dBf[0] += dF;

            double dCt = dC * i[t] * (1.0 - cHat[t] * cHat[t]);
            dWc[t] += dCt * x[t];
            dBc[0] += dCt;

            double dXt = (dI * wi[t] + dF * wf[t] + dCt * wc[t] + dO * wo[t]);
            dHNext[t] = dXt;

            // Clip gradients here
            const double gradientClipThreshold = 1.0; // Adjust as needed
            if (dWo[t] > gradientClipThreshold) dWo[t] = gradientClipThreshold;
            if (dWo[t] < -gradientClipThreshold) dWo[t] = -gradientClipThreshold;
            if (dWi[t] > gradientClipThreshold) dWi[t] = gradientClipThreshold;
            if (dWi[t] < -gradientClipThreshold) dWi[t] = -gradientClipThreshold;
            if (dWf[t] > gradientClipThreshold) dWf[t] = gradientClipThreshold;
            if (dWf[t] < -gradientClipThreshold) dWf[t] = -gradientClipThreshold;
            if (dWc[t] > gradientClipThreshold) dWc[t] = gradientClipThreshold;
            if (dWc[t] < -gradientClipThreshold) dWc[t] = -gradientClipThreshold;
            if (dBo[0] > gradientClipThreshold) dBo[0] = gradientClipThreshold;
            if (dBo[0] < -gradientClipThreshold) dBo[0] = -gradientClipThreshold;
            if (dBi[0] > gradientClipThreshold) dBi[0] = gradientClipThreshold;
            if (dBi[0] < -gradientClipThreshold) dBi[0] = -gradientClipThreshold;
            if (dBf[0] > gradientClipThreshold) dBf[0] = gradientClipThreshold;
            if (dBf[0] < -gradientClipThreshold) dBf[0] = -gradientClipThreshold;
            if (dBc[0] > gradientClipThreshold) dBc[0] = gradientClipThreshold;
            if (dBc[0] < -gradientClipThreshold) dBc[0] = -gradientClipThreshold;


            // Update weights and biases with gradient descent
            wo[t] -= learningRate * dWo[t];
            bo[0] -= learningRate * dBo[0];
            wi[t] -= learningRate * dWi[t];
            bi[0] -= learningRate * dBi[0];
            wf[t] -= learningRate * dWf[t];
            bf[0] -= learningRate * dBf[0];
            wc[t] -= learningRate * dWc[t];
            bc[0] -= learningRate * dBc[0];
        }
    }

    // Get output
    std::vector<double> getOutput() {
        return h;
    }

    // Get cell state
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

    // Getter for the gradient of the hidden state
    std::vector<double> getGradientHiddenState() {
        return h;
    }

    // Getter for the gradient of the cell state
    std::vector<double> getGradientCellState() {
        return c;
    }
};

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;

    // Cell parameters
    int inputSize;
    int hiddenSize;
    int outputSize;

public:
    LSTMNetwork(int inputSize, int hiddenSize, int outputSize) : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        // initialise LSTM cells
        LSTMCell cell(inputSize, hiddenSize);
        cells.push_back(cell);
    }

    void train(std::vector<std::vector<double>>& inputs, std::vector<int> targets) {
        // Train LSTM network
        for (int i = 0; i < inputs.size(); i++) {
            forward(inputs[i]);
            std::cout << "Forward pass " << i << "/" << inputs.size() << " complete" << std::endl;

            // Compute loss for this time step
            double loss = lossFunction(targets[i], cells.back().getOutput()[0]);

            std::cout << "Loss: " << loss << std::endl;

            // Backward pass through time for each LSTM cell
            BPTT(inputs, targets);
        }
    }

    // Update the forward call in the LSTMNetwork forward method.
    void forward(std::vector<double>& inputs) {
        for (int i = 0; i < cells.size(); i++) {
            cells[i].forward(inputs);
        }
    }

    void BPTT(std::vector<std::vector<double>>& inputs, std::vector<int> targets) {
        // Backward pass through time

        // Initialize gradients for weights and biases to zero
        for (LSTMCell& cell : cells) {
            cell.initialiseGradients();
        }

        std::cout << "Initialised gradients" << std::endl;

        // Loop over input sequences in reverse order
        for (int seqIndex = inputs.size() - 1; seqIndex >= 0; seqIndex--) {
            // Forward pass
            forward(inputs[seqIndex]);

            std::cout << "Forward pass " << inputs.size() - seqIndex << "/" << inputs.size() << " complete" << std::endl;

            std::cout << "Output: " << cells.back().getOutput()[0] << std::endl;

            // Compute loss for this time step
            double loss = lossFunction(targets[seqIndex], cells.back().getOutput()[0]);

            std::cout << "Loss: " << loss << std::endl;

            // Backward pass through time for each LSTM cell
            std::vector<double> dNextHiddenState(hiddenSize, 0.0);
            std::vector<double> dNextCellState(hiddenSize, 0.0);

            for (int i = cells.size() - 1; i >= 0; i--) {
                LSTMCell& cell = cells[i];

                // Compute gradients for this time step and accumulate them
                cell.backward(dNextHiddenState, dNextCellState);

                std::cout << "Backward pass " << i << "/" << cells.size() << " complete" << std::endl;

                // Update the gradients for the next time step
                dNextHiddenState = cell.getGradientHiddenState();
                dNextCellState = cell.getGradientCellState();
            }
        }
    }

    double lossFunction(int target, int prediction) {
        // Calculate loss
        double loss = 0.5 * pow(target - prediction, 2);
        return loss;
    }

    // double lossFunction(int target, double prediction) {
    //     return -((target * log(prediction)) + ((1 - target) * log(1 - prediction)));
    // }


    void eval(const std::vector<std::vector<double>>& inputs, std::vector<int> targets) {
        // Evaluate LSTM network
        std::vector<double> predictions;
        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> input = inputs[i];
            forward(input);
            // std::vector<double> output = cells[0].getOutput();
            std::vector<double> output = cells.back().getOutput();
            predictions.push_back(output[0]);
        }

        double totalLoss = 0;
        for (int i = 0; i < predictions.size(); i++) {
            double lossValue = lossFunction(targets[i], predictions[i]);
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

    // initialise LSTM network
    LSTMNetwork network(inputSize, hiddenSize, outputSize);

    // Train LSTM network
    // for (int i = 0; i < train_input_data.size(); i++) {
    //     network.forward(train_input_data[i]);
    //     std::cout << "Forward pass " << i << "/" << train_input_data.size() << " complete" << std::endl;
    // }
    // for (int i = 0; i < 100; i++) {
    //     network.BPTT(train_input_data, train_target_data);
    //     std::cout << "Backpropagation through time " << i << "/100 complete" << std::endl;
    // }

    network.train(train_input_data, train_target_data);

    std::cout << "Training complete" << std::endl;

    network.eval(test_input_data, test_target_data);

    std::cout << "Testing complete" << std::endl;

    return 0;
}
