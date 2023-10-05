#include <iostream>
#include <valarray>
#include <vector>
#include <memory>
#include <numeric>
#include <stdlib.h>
#include <math.h>

#include "data.hpp"

class Sigmoid {
 public:
  Sigmoid(int logit_size);
  float Logit(float p) const;
  static float Logistic(float p);

 private:
  float SlowLogit(float p);
  int logit_size_;
  std::vector<float> logit_table_;
};

Sigmoid::Sigmoid(int logit_size) : logit_size_(logit_size),
    logit_table_(logit_size, 0) {
  for (int i = 0; i < logit_size_; ++i) {
    logit_table_[i] = SlowLogit((i + 0.5) / logit_size_);
  }
}

float Sigmoid::Logit(float p) const {
  return logit_table_[p * logit_size_];
}

float Sigmoid::Logistic(float p) {
  return 1 / (1 + exp(-p));
}

float Sigmoid::SlowLogit(float p) {
  return log(p / (1 - p));
}

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float learning_rate, float gradient_clip);
  void ForwardPass(const std::valarray<float>& input, int input_symbol,
      std::valarray<float>* hidden, int hidden_start);
  void BackwardPass(const std::valarray<float>& input, int epoch,
      int layer, int input_symbol, std::valarray<float>* hidden_error);
  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

 private:
  std::valarray<float> state_, output_gate_error_, state_error_,
      input_node_error_, input_gate_error_, forget_gate_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, output_gate_state_,
      input_node_state_, input_gate_state_, forget_gate_state_, last_state_,
      forget_gate_, input_node_, input_gate_, output_gate_, forget_gate_update_,
      input_node_update_, input_gate_update_, output_gate_update_;
  float learning_rate_, gradient_clip_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;

  void ClipGradients(std::valarray<float>* arr);
};

LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float learning_rate, float gradient_clip) : state_(num_cells),
    output_gate_error_(num_cells), state_error_(num_cells),
    input_node_error_(num_cells), input_gate_error_(num_cells),
    forget_gate_error_(num_cells), stored_error_(num_cells),
    tanh_state_(std::valarray<float>(num_cells), horizon),
    output_gate_state_(std::valarray<float>(num_cells), horizon),
    input_node_state_(std::valarray<float>(num_cells), horizon),
    input_gate_state_(std::valarray<float>(num_cells), horizon),
    forget_gate_state_(std::valarray<float>(num_cells), horizon),
    last_state_(std::valarray<float>(num_cells), horizon),
    forget_gate_(std::valarray<float>(input_size), num_cells),
    input_node_(std::valarray<float>(input_size), num_cells),
    input_gate_(std::valarray<float>(input_size), num_cells),
    output_gate_(std::valarray<float>(input_size), num_cells),
    forget_gate_update_(std::valarray<float>(input_size), num_cells),
    input_node_update_(std::valarray<float>(input_size), num_cells),
    input_gate_update_(std::valarray<float>(input_size), num_cells),
    output_gate_update_(std::valarray<float>(input_size), num_cells),
    learning_rate_(learning_rate), gradient_clip_(gradient_clip),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size) {
  float low = -0.2;
  float range = 0.4;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_[i].size(); ++j) {
      forget_gate_[i][j] = low + Rand() * range;
      input_node_[i][j] = low + Rand() * range;
      input_gate_[i][j] = low + Rand() * range;
      output_gate_[i][j] = low + Rand() * range;
    }
    forget_gate_[i][forget_gate_[i].size() - 1] = 1;
  }
}

void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
    std::valarray<float>* hidden, int hidden_start) {
  last_state_[epoch_] = state_;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_state_[epoch_][i] = Sigmoid::Logistic(std::inner_product(
        &input[0], &input[input.size()],
        &forget_gate_[i][output_size_], forget_gate_[i][input_symbol]));
    input_node_state_[epoch_][i] = tanh(std::inner_product(&input[0],
        &input[input.size()], &input_node_[i][output_size_],
        input_node_[i][input_symbol]));
    input_gate_state_[epoch_][i] = Sigmoid::Logistic(std::inner_product(
        &input[0], &input[input.size()],
        &input_gate_[i][output_size_], input_gate_[i][input_symbol]));
    output_gate_state_[epoch_][i] = Sigmoid::Logistic(std::inner_product(
        &input[0], &input[input.size()],
        &output_gate_[i][output_size_], output_gate_[i][input_symbol]));
  }
  state_ *= forget_gate_state_[epoch_];
  state_ += input_node_state_[epoch_] * input_gate_state_[epoch_];
  tanh_state_[epoch_] = tanh(state_);
  std::slice slice = std::slice(hidden_start, num_cells_, 1);
  (*hidden)[slice] = output_gate_state_[epoch_] * tanh_state_[epoch_];
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
  }
}

void LstmLayer::BackwardPass(const std::valarray<float>&input, int epoch,
    int layer, int input_symbol, std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      forget_gate_update_[i] = 0;
      input_node_update_[i] = 0;
      input_gate_update_[i] = 0;
      output_gate_update_[i] = 0;
    }
  } else {
    stored_error_ += *hidden_error;
  }

  output_gate_error_ = tanh_state_[epoch] * stored_error_ *
      output_gate_state_[epoch] * (1.0f - output_gate_state_[epoch]);
  state_error_ += stored_error_ * output_gate_state_[epoch] * (1.0f -
      (tanh_state_[epoch] * tanh_state_[epoch]));
  input_node_error_ = state_error_ * input_gate_state_[epoch] * (1.0f -
      (input_node_state_[epoch] * input_node_state_[epoch]));
  input_gate_error_ = state_error_ * input_node_state_[epoch] *
      input_gate_state_[epoch] * (1.0f - input_gate_state_[epoch]);
  forget_gate_error_ = state_error_ * last_state_[epoch] *
      forget_gate_state_[epoch] * (1.0f - forget_gate_state_[epoch]);

  *hidden_error = 0;
  if (layer > 0) {
    int offset = output_size_ + num_cells_ + input_size_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      for (unsigned int j = offset; j < offset + num_cells_; ++j) {
        (*hidden_error)[j-offset] += input_node_[i][j] * input_node_error_[i];
        (*hidden_error)[j-offset] += input_gate_[i][j] * input_gate_error_[i];
        (*hidden_error)[j-offset] += forget_gate_[i][j] * forget_gate_error_[i];
        (*hidden_error)[j-offset] += output_gate_[i][j] * output_gate_error_[i];
      }
    }
  }

  if (epoch > 0) {
    state_error_ *= forget_gate_state_[epoch];
    stored_error_ = 0;
    int offset = output_size_ + input_size_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      for (unsigned int j = offset; j < offset + num_cells_; ++j) {
        stored_error_[j-offset] += input_node_[i][j] * input_node_error_[i];
        stored_error_[j-offset] += input_gate_[i][j] * input_gate_error_[i];
        stored_error_[j-offset] += forget_gate_[i][j] * forget_gate_error_[i];
        stored_error_[j-offset] += output_gate_[i][j] * output_gate_error_[i];
      }
    }
  }

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);

  std::slice slice = std::slice(output_size_, input.size(), 1);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_update_[i][slice] += (learning_rate_ * forget_gate_error_[i]) *
        input;
    input_node_update_[i][slice] += (learning_rate_ * input_node_error_[i]) *
        input;
    input_gate_update_[i][slice] += (learning_rate_ * input_gate_error_[i]) *
        input;
    output_gate_update_[i][slice] += (learning_rate_ * output_gate_error_[i]) *
        input;
    forget_gate_update_[i][input_symbol] += learning_rate_ *
        forget_gate_error_[i];
    input_node_update_[i][input_symbol] += learning_rate_ *
        input_node_error_[i];
    input_gate_update_[i][input_symbol] += learning_rate_ *
        input_gate_error_[i];
    output_gate_update_[i][input_symbol] += learning_rate_ *
        output_gate_error_[i];
  }
  if (epoch == 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
        forget_gate_[i] += forget_gate_update_[i];
        input_node_[i] += input_node_update_[i];
        input_gate_[i] += input_gate_update_[i];
        output_gate_[i] += output_gate_update_[i];
    }
  }
}

class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  std::valarray<float>& Perceive(unsigned int input);
  std::valarray<float>& Predict(unsigned int input);
  void SetInput(int index, float val);

 private:
  std::vector<std::unique_ptr<LstmLayer>> layers_;
  std::vector<unsigned int> input_history_;
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_,
      output_layer_;
  std::valarray<std::valarray<float>> output_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
};

class ByteModel {
 public:
  ByteModel(const std::vector<bool>& vocab, Lstm* lstm);
  float Predict();
  void Perceive(int bit);
  std::valarray<float> getProbs() { return probs_; } // Getter for probs_

 protected:
  int top_, mid_, bot_;
  std::valarray<int> byte_map_;
  std::valarray<float> probs_;
  unsigned int bit_context_;
  std::unique_ptr<Lstm> lstm_;
  const std::vector<bool>& vocab_;
};

ByteModel::ByteModel(const std::vector<bool>& vocab, Lstm* lstm)
    : top_(128), mid_(0), bot_(0), byte_map_(256), probs_(256),
    bit_context_(0), lstm_(lstm), vocab_(vocab) {
  for (int i = 0; i < 256; ++i) {
    byte_map_[i] = i;
    probs_[i] = 0;
  }
}

float ByteModel::Predict() {

    for (int i = 0; i < probs_.size(); ++i) {
        std::cout << probs_[i] << " ";
    }
    std::cout << std::endl;

    float total = 0;
    for (int i = 0; i < 256; ++i) {
        probs_[i] = 0;
    }
    for (int i = 0; i < 256; ++i) {
        int index = top_ + mid_ + bot_ + i;
        lstm_->SetInput(0, static_cast<float>(index) / 256.0);
        probs_[i] = lstm_->Predict(0)[!vocab_.empty() ? 1 : 0]; // Changed
        total += probs_[i];
    }
    if (total == 0) total = 1;
    for (int i = 0; i < 256; ++i) {
        probs_[i] /= total;
    }
    return probs_[byte_map_[top_]];
}

void ByteModel::Perceive(int bit) {
  int index = top_ + mid_ + bot_;
  lstm_->SetInput(0, static_cast<float>(index) / 256.0);
  lstm_->SetInput(1, static_cast<float>(bit));
  lstm_->Perceive(0);
  if (bit == 0) {
    mid_ = (mid_ * 2) + 1;
  } else {
    mid_ = (mid_ * 2);
  }
  if (mid_ > 127) {
    mid_ = 0;
    bot_ = (bot_ * 2) + 1;
  } else {
    bot_ = (bot_ * 2);
  }
  if (bot_ > 127) {
    bot_ = 0;
    top_ = (top_ * 2) + 1;
  } else {
    top_ = (top_ * 2);
  }
  bit_context_ = (bit_context_ << 1) | bit;
  if (bit_context_ > 255) {
    bit_context_ = 0;
  }
}

Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
    layer_input_(std::valarray<std::valarray<float>>(std::valarray<float>
    (input_size + 1 + num_cells * 2), num_layers), horizon),
    output_layer_(std::valarray<std::valarray<float>>(std::valarray<float>
    (num_cells * num_layers + 1), output_size), horizon),
    output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), input_size_(input_size), output_size_(output_size) {
  srand(69761);
  hidden_[hidden_.size() - 1] = 1;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    layer_input_[epoch][0].resize(1 + num_cells + input_size);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.push_back(std::unique_ptr<LstmLayer>(new LstmLayer(
        layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, learning_rate, gradient_clip)));
  }
}

void Lstm::SetInput(int index, float val) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    layer_input_[epoch_][i][index] = val;
  }
}

std::valarray<float>& Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = 0;
          if (i == input_history_[epoch]) error = (1 - output_[epoch][i]);
          else error = -output_[epoch][i];
          for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
          }
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer]->BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_);
      }
    }
  }

  output_layer_[epoch_] = output_layer_[last_epoch];
  for (unsigned int i = 0; i < output_size_; ++i) {
    float error = 0;
    if (i == input) error = (1 - output_[last_epoch][i]);
    else error = -output_[last_epoch][i];
    output_layer_[epoch_][i] += learning_rate_ * error * hidden_;
  }
  return Predict(input);
}

std::valarray<float>& Lstm::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_, begin(layer_input_[epoch_][i]) +
        input_size_);
    layers_[i]->ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
        num_cells_);
    if (i < layers_.size() - 1) {
      auto start2 = begin(layer_input_[epoch_][i + 1]) + num_cells_ +
          input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
  float max_out = 0;
  for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = 0;
    for (unsigned int j = 0; j < hidden_.size(); ++j) {
      sum += hidden_[j] * output_layer_[epoch_][i][j];
    }
    output_[epoch_][i] = sum;
    max_out = std::max(sum, max_out);
  }
  for (unsigned int i = 0; i < output_size_; ++i) {
    output_[epoch_][i] = exp(output_[epoch_][i] - max_out);
  }
  output_[epoch_] /= output_[epoch_].sum();
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  return output_[epoch];
}

class BytePredictor {
 public:
  BytePredictor(const std::vector<bool>& vocab);
  int Predict();
  void Update(int bit);

 private:
  std::vector<bool> vocab_;
  ByteModel model_;
};

BytePredictor::BytePredictor(const std::vector<bool>& vocab) : vocab_(vocab),
    model_(vocab_, new Lstm(1, 2, 2, 1, 4, 0.1, 0.1)) {
}

int BytePredictor::Predict() {
    model_.Predict();
    if (model_.getProbs()[0] > 0.5) return 0;
    return 1;
}

void BytePredictor::Update(int bit) {
    model_.Perceive(bit);
}

int main() {
    std::vector<bool> vocab = {false, true};
    BytePredictor predictor(vocab);

    // Load data from emails.csv and split into train and test sets
    std::vector<std::vector<double>> input_data;
    std::vector<int> target_data;
    std::vector<std::string> header;
    loadData("./inc/emailsHotEncoding.csv", input_data, target_data, header, "Prediction", 250, 10);

    // Train-test split, 80% train, 20% test
    std::vector<std::vector<double>> train_input_data;
    std::vector<int> train_target_data;
    std::vector<std::vector<double>> test_input_data;
    std::vector<int> test_target_data;
    trainTestSplit(input_data, target_data, train_input_data, train_target_data, test_input_data, test_target_data);

    int correct_predictions = 0;
    int total_predictions = 0;

    std::cout << "Starting Training Phase" << std::endl;

    // Training phase
    for (int i = 0; i < train_input_data.size(); ++i) {
        for (int j = 0; j < train_input_data[i].size(); ++j) {
            std::cout << "Training example " << i+1 << " bit " << j+1 << std::endl;
            int bit = static_cast<int>(train_input_data[i][j]);
            int prediction = predictor.Predict();
            std::cout << "Prediction: " << prediction << std::endl;
            predictor.Update(bit);
            if (prediction == train_target_data[i]) {
                correct_predictions++;
            }
            total_predictions++;
        }
        std::cout << "Completed " << i+1 << "th training example, out of " << train_input_data.size() << std::endl;
    }

    std::cout << "Completed Training Phase" << std::endl;

    // Testing phase
    for (int i = 0; i < test_input_data.size(); ++i) {
        for (int j = 0; j < test_input_data[i].size(); ++j) {
        int bit = static_cast<int>(test_input_data[i][j]);
        int prediction = predictor.Predict();
        if (prediction == test_target_data[i]) {
            correct_predictions++;
        }
        total_predictions++;
        }
    }

    std::cout << "Completed Testing Phase" << std::endl;

    double accuracy = static_cast<double>(correct_predictions) / total_predictions * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
