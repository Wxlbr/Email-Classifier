#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>  // for random weight initialization

// Define the sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Define the ReLU activation function
double relu(double x) {
    // std::cout << "Relu " << x << " " << std::max(0.0, x) << std::endl;
    return std::max(0.0, x);
}

// Define a TCN layer class
class TCNLayer {
public:
    TCNLayer(int input_size, int output_size) {
        input_size_ = input_size;
        output_size_ = output_size;
        weights_.resize(input_size_, std::vector<double>(output_size_, 0.0));
        biases_.resize(output_size_, 0.0);
        // Initialize weights with random values
        srand(time(0));
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < output_size_; ++j) {
                weights_[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }
    }

    // Forward pass through the TCN layer
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output(output_size_, 0.0);
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                output[i] += input[j] * weights_[j][i];
            }
            output[i] += biases_[i];
            output[i] = relu(output[i]);  // Apply ReLU activation
        }
        return output;
    }

    // Update the weights and biases using backpropagation
    void backward(const std::vector<double>& input, const std::vector<double>& d_output, double learning_rate) {
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                double gradient = input[j] * d_output[i];
                weights_[j][i] -= learning_rate * gradient;
            }
            biases_[i] -= learning_rate * d_output[i];
        }
    }

private:
    int input_size_;
    int output_size_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
};

// Define a TCN model class
class TCNModel {
public:
    TCNModel(int input_size, std::vector<int> hidden_sizes) {
        layers_.emplace_back(input_size, hidden_sizes[0]);

        for (int i = 1; i < hidden_sizes.size(); ++i) {
            layers_.emplace_back(hidden_sizes[i - 1], hidden_sizes[i]);
        }

        output_layer_.emplace_back(hidden_sizes.back(), 1);
    }

    // Forward pass through the TCN model
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> x = input;
        for (auto& layer : layers_) {
            x = layer.forward(x);
        }
        return output_layer_[0].forward(x);  // Final output
    }

    // Train the TCN model using backpropagation
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            double total_accuracy = 0.0;

            for (int i = 0; i < inputs.size(); ++i) {
                const std::vector<double>& input = inputs[i];
                double target = targets[i];

                // Forward pass
                double output = forward(input)[0];

                // Compute loss
                double loss = 0.5 * pow(output - target, 2);
                total_loss += loss;

                // Compute accuracy
                double accuracy = (fabs(output - target) < 0.5) ? 1.0 : 0.0;
                total_accuracy += accuracy;

                // Backpropagation
                double d_output = output - target;
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                    const std::vector<double>& layer_input = (it == layers_.rbegin()) ? input : it->forward(layer_input);
                    it->backward(layer_input, std::vector<double>{d_output}, learning_rate);
                }
            }

            // Print loss and accuracy after each epoch
            total_loss /= inputs.size();
            total_accuracy /= inputs.size();
            std::cout << "Epoch " << epoch + 1 << ": Loss = " << total_loss << ", Accuracy = " << total_accuracy << std::endl;
        }
    }

    // Predict the output for a given input
    std::vector<double> predict(const std::vector<double>& input) {
        return forward(input);
    }

private:
    std::vector<TCNLayer> layers_;
    std::vector<TCNLayer> output_layer_;
};