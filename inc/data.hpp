#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

void loadData(std::string filename, std::vector<std::vector<double>>& data, std::vector<int>& labels, std::vector<std::string>& header, std::string flagHeader, int maxRows, int maxColumns) {
    // Load data from emails.csv
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;
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
        std::cerr << "Flag header not found: " << flagHeader << std::endl;
        return;
    }

    std::vector<std::vector<double>> raw_data;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
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

        row.resize(maxColumns);
        raw_data.push_back(row);
    }

    raw_data.resize(maxRows);

    // Min-Max Scaling
    for (size_t col = 0; col < raw_data[0].size(); ++col) {
        double min_val = raw_data[0][col];
        double max_val = raw_data[0][col];

        // Find min and max values in the column
        for (size_t row = 0; row < raw_data.size(); ++row) {
            if (raw_data[row][col] < min_val) {
                min_val = raw_data[row][col];
            }
            if (raw_data[row][col] > max_val) {
                max_val = raw_data[row][col];
            }
        }

        // Apply Min-Max scaling to the column
        for (size_t row = 0; row < raw_data.size(); ++row) {
            if (max_val != min_val) {
                raw_data[row][col] = (raw_data[row][col] - min_val) / (max_val - min_val);
            }
            else {
                // Handle the case where min_val and max_val are the same (constant column)
                raw_data[row][col] = max_val; // You can choose an appropriate value
            }
        }

        // What range do you expect the data to be in? A: 0-1
    }

    // Copy the scaled data to the output data vector
    data = raw_data;
}


void loadTextData(std::string filename, std::vector<std::vector<std::string>>& data, std::vector<int>& labels, std::vector<std::string>& header, std::string flagHeader) {
    // flagHeader = "label_num"

    // Temp Variables
    std::vector<int> ignoreIndices = {0, 1};

    // Load data from emails.csv
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;
    int flagIndex = -1;
    int index = 0;

    while (getline(ss, cell, ',')) {
        if (std::find(ignoreIndices.begin(), ignoreIndices.end(), index) != ignoreIndices.end()) {
            index++;
            continue;
        }
        if (cell == flagHeader) {
            flagIndex = index;
        }
        else {
            header.push_back(cell);
        }
        index++;
    }

    if (flagIndex == -1) {
        std::cerr << "Flag header not found: " << flagHeader << std::endl;
        return;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        int index = 0;

        while (getline(ss, cell, ',')) {
            if (std::find(ignoreIndices.begin(), ignoreIndices.end(), index) != ignoreIndices.end()) {
                index++;
                continue;
            }
            if (index == flagIndex) {
                labels.push_back(stoi(cell));
            }
            else {
                row.push_back(cell);
            }
            index++;
        }
    }
}

void trainTestSplit(std::vector<std::vector<double>>& input_data, std::vector<int>& target_data,
        std::vector<std::vector<double>>& train_input_data, std::vector<int>& train_target_data,
        std::vector<std::vector<double>>& test_input_data, std::vector<int>& test_target_data) {
    // Train test split, 80% train, 20% test
    size_t train_size = (size_t)(input_data.size() * 0.8);

    // Split into spam and ham
    std::vector<std::vector<double>> spam_input_data;
    std::vector<int> spam_target_data;

    std::vector<std::vector<double>> ham_input_data;
    std::vector<int> ham_target_data;

    for (size_t i = 0; i < input_data.size(); ++i) {
        if (target_data[i] == 1) {
            spam_input_data.push_back(input_data[i]);
            spam_target_data.push_back(target_data[i]);
        }
        else {
            ham_input_data.push_back(input_data[i]);
            ham_target_data.push_back(target_data[i]);
        }
    }

    // Split spam into train and test
    size_t spam_train_size = (size_t)(spam_input_data.size() * 0.8);

    for (size_t i = 0; i < spam_train_size; ++i) {
        train_input_data.push_back(spam_input_data[i]);
        train_target_data.push_back(spam_target_data[i]);
    }

    for (size_t i = spam_train_size; i < spam_input_data.size(); ++i) {
        test_input_data.push_back(spam_input_data[i]);
        test_target_data.push_back(spam_target_data[i]);
    }

    // Split ham into train and test
    size_t ham_train_size = (size_t)(ham_input_data.size() * 0.8);

    for (size_t i = 0; i < ham_train_size; ++i) {
        train_input_data.push_back(ham_input_data[i]);
        train_target_data.push_back(ham_target_data[i]);
    }

    for (size_t i = ham_train_size; i < ham_input_data.size(); ++i) {
        test_input_data.push_back(ham_input_data[i]);
        test_target_data.push_back(ham_target_data[i]);
    }


    // for (int i = 0; i < train_size; ++i) {
    //     train_input_data.push_back(input_data[i]);
    //     train_target_data.push_back(target_data[i]);
    // }

    // for (int i = train_size; i < input_data.size(); ++i) {
    //     test_input_data.push_back(input_data[i]);
    //     test_target_data.push_back(target_data[i]);
    // }
}