#include <iostream>
#include <vector>
#include <cmath>

#include "data.hpp"

class LSTMCell {
private:
    std::vector<double> input_gate_weights;
    double input_gate_bias;
    std::vector<double> forget_gate_weights;
    double forget_gate_bias;
    std::vector<double> output_gate_weights;
    double output_gate_bias;
    std::vector<double> cell_state_weights;
    double cell_state_bias;

    std::vector<double> cell_state;
    std::vector<double> hidden_state;

    std::vector<double> input;
    std::vector<double> predicted_output;

    std::vector<double> output_gate;

    double learning_rate = 0.01;

public:
    LSTMCell(int input_size) {
        // Initialize weights and biases
        input_gate_weights.resize(input_size);
        forget_gate_weights.resize(input_size);
        output_gate_weights.resize(input_size);
        cell_state_weights.resize(input_size);

        for (int i = 0; i < input_size; ++i) {
            input_gate_weights[i] = 0.5;
            forget_gate_weights[i] = 0.6;
            output_gate_weights[i] = 0.7;
            cell_state_weights[i] = 0.8;
        }

        input_gate_bias = 0.1;
        forget_gate_bias = 0.2;
        output_gate_bias = 0.3;
        cell_state_bias = 0.4;

        // Initialize cell state and hidden state
        cell_state.resize(input_size, 0.0);
        hidden_state.resize(input_size, 0.0);
        input.resize(input_size, 0.0);
        predicted_output.resize(input_size, 0.0);
    }

    void forward(const std::vector<double>& input_values) {
        if (input_values.size() != input_gate_weights.size()) {
            std::cerr << "Input size does not match weight size." << std::endl;
            return;
        }

        input = input_values;

        // Input gate
        std::vector<double> input_gate(input_values.size());
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_gate[i] = sigmoid(input_gate_weights[i] * input[i] + input_gate_bias);
        }

        // Forget gate
        std::vector<double> forget_gate(input_values.size());
        for (size_t i = 0; i < input_values.size(); ++i) {
            forget_gate[i] = sigmoid(forget_gate_weights[i] * input[i] + forget_gate_bias);
        }

        // Update cell state
        for (size_t i = 0; i < input_values.size(); ++i) {
            cell_state[i] = cell_state[i] * forget_gate[i] + input_gate[i] * tanh(cell_state_weights[i] * input[i] + cell_state_bias);
        }

        // Output gate
        output_gate.resize(input_values.size()); // Initialize the output_gate vector
        for (size_t i = 0; i < input_values.size(); ++i) {
            output_gate[i] = sigmoid(output_gate_weights[i] * input[i] + output_gate_bias);
        }


        // Update hidden state
        for (size_t i = 0; i < input_values.size(); ++i) {
            hidden_state[i] = output_gate[i] * tanh(cell_state[i]);
        }

        // Predicted output (for simplicity, it's the same as the hidden state)
        predicted_output = hidden_state;
    }

    std::vector<double> getHiddenState() const {
        return hidden_state;
    }

    std::vector<double> getPredictedOutput() const {
        return predicted_output;
    }

    std::vector<double> getInput() const {
        return input;
    }

    void backward(const std::vector<double>& delta_cell_state, const std::vector<double>& delta_hidden_state) {
        if (delta_cell_state.size() != delta_hidden_state.size() || delta_cell_state.size() != input.size()) {
            std::cerr << "Delta sizes do not match input size." << std::endl;
            return;
        }

        // Backpropagate through time (BPTT)

        // Input gate, forget gate, and output gate gradients
        std::vector<double> delta_input_gate(input.size());
        std::vector<double> delta_forget_gate(input.size());
        std::vector<double> delta_output_gate(input.size());

        for (size_t i = 0; i < input.size(); ++i) {
            // Input gate
            delta_input_gate[i] = delta_cell_state[i] * tanh(cell_state_weights[i] * input[i] + cell_state_bias) * sigmoid_derivative(input_gate_weights[i] * input[i] + input_gate_bias);

            // Forget gate
            delta_forget_gate[i] = delta_cell_state[i] * cell_state[i] * sigmoid_derivative(forget_gate_weights[i] * input[i] + forget_gate_bias);

            // Output gate
            delta_output_gate[i] = delta_hidden_state[i] * tanh(cell_state[i]) * sigmoid_derivative(output_gate_weights[i] * input[i] + output_gate_bias);
        }

        // Gradient for cell state
        std::vector<double> delta_cell_state_total(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            delta_cell_state_total[i] = delta_cell_state[i] + delta_hidden_state[i] * output_gate[i] * (1 - tanh(cell_state[i]) * tanh(cell_state[i]));
        }

        // Gradient for input
        std::vector<double> delta_input(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            delta_input[i] = (delta_input_gate[i] + delta_forget_gate[i]) * input_gate_weights[i];
        }

        // Update weights and biases
        for (size_t i = 0; i < input.size(); ++i) {
            input_gate_weights[i] -= learning_rate * delta_input_gate[i] * input[i];
            forget_gate_weights[i] -= learning_rate * delta_forget_gate[i] * input[i];
            output_gate_weights[i] -= learning_rate * delta_output_gate[i] * input[i];
            cell_state_weights[i] -= learning_rate * delta_cell_state_total[i] * input[i];
        }

        // Update biases
        input_gate_bias -= learning_rate * sum(delta_input_gate);
        forget_gate_bias -= learning_rate * sum(delta_forget_gate);
        output_gate_bias -= learning_rate * sum(delta_output_gate);
        cell_state_bias -= learning_rate * sum(delta_cell_state_total);
    }

private:
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double tanh(double x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    double sum(const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum;
    }
};

class LSTMNetwork {
private:
    std::vector<LSTMCell> cells;

public:
    LSTMNetwork(int num_cells, int input_size) {
        cells.resize(num_cells, LSTMCell(input_size));
    }

    void train(const std::vector<std::vector<double>>& input_sequence, const std::vector<int>& target_sequence) {
        if (input_sequence.size() != target_sequence.size()) {
            std::cerr << "Input and target sequences must have the same length." << std::endl;
            return;
        }

        // Initialize delta_cell_state and delta_hidden_state for each cell
        std::vector<std::vector<double>> delta_cell_state(cells.size());
        std::vector<std::vector<double>> delta_hidden_state(cells.size());

        for (size_t t = 0; t < input_sequence.size(); ++t) {
            // Initialize delta_cell_state and delta_hidden_state for this time step
            for (size_t i = 0; i < cells.size(); ++i) {
                delta_cell_state[i].resize(cells[i].getInput().size(), 0.0);
                delta_hidden_state[i].resize(cells[i].getInput().size(), 0.0);
            }

            // Forward pass
            for (size_t i = 0; i < cells.size(); ++i) {
                cells[i].forward(input_sequence[t]);
            }

            // Calculate the error at the output layer (for classification)
            std::vector<std::vector<double>> output_error(cells.size());
            for (size_t i = 0; i < cells.size(); ++i) {
                for (size_t j = 0; j < cells[i].getPredictedOutput().size(); ++j) {
                    double target = (j == target_sequence[t]) ? 1.0 : 0.0; // One-hot encoding
                    output_error[i].push_back(cells[i].getPredictedOutput()[j] - target);
                }
            }

            // Backpropagate the error through the LSTM cells
            for (size_t i = 0; i < cells.size(); ++i) {
                delta_hidden_state[i] = output_error[i];
                cells[i].backward(delta_cell_state[i], delta_hidden_state[i]);
            }
        }
    }


    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& input_sequence) {
        std::vector<std::vector<double>> predictions;

        for (const std::vector<double>& input_values : input_sequence) {
            std::vector<double> prediction(cells.size());
            for (size_t i = 0; i < cells.size(); ++i) {
                cells[i].forward(input_values);
                prediction[i] = cells[i].getPredictedOutput()[0]; // TODO: [0]
            }
            predictions.push_back(prediction);
        }

        return predictions;
    }
};

int main() {
    // Create an LSTM network with 3 cells, each with an input size of 100
    int num_cells = 3;

    int input_sequence_size = 1000;
    int input_length = 100;

    LSTMNetwork lstm(num_cells, input_length);

    // Load data from emails.csv
    std::vector<std::vector<double>> input_sequence;
    std::vector<int> target_sequence;
    std::vector<std::string> header;

    loadData("./inc/emailsHotEncoding.csv", input_sequence, target_sequence, header, "Prediction", input_sequence_size, input_length);

    // train test split
    std::vector<std::vector<double>> input_sequence_train;
    std::vector<int> target_sequence_train;
    std::vector<std::vector<double>> input_sequence_test;
    std::vector<int> target_sequence_test;

    trainTestSplit(input_sequence, target_sequence, input_sequence_train, target_sequence_train, input_sequence_test, target_sequence_test);

    // Train the LSTM network
    lstm.train(input_sequence_train, target_sequence_train);

    // Make predictions
    std::vector<std::vector<double>> predictions = lstm.predict(input_sequence_test);

    int correct = 0;

    // Print predictions
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < predictions.size(); i++) {
        double pred = predictions[i][0];
        int pred_class = (pred > 0.5) ? 1 : 0;
        std::cout << "Prediction: " << pred << " Class: " << pred_class << " Target: " << target_sequence_test[i] << std::endl;
        if (pred_class == target_sequence_test[i]) {
            correct++;
        }
    }

    std::cout << "Accuracy: " << (double) correct / predictions.size() << std::endl;

    return 0;
}