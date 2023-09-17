#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

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

        row.resize(maxColumns); // Shrink row to first x elements
        data.push_back(row);
    }

    data.resize(maxRows); // Shrink data to first x rows
}

void trainTestSplit(std::vector<std::vector<double>>& input_data, std::vector<int>& target_data,
        std::vector<std::vector<double>>& train_input_data, std::vector<int>& train_target_data,
        std::vector<std::vector<double>>& test_input_data, std::vector<int>& test_target_data) {
    // Train test split, 80% train, 20% test
    size_t train_size = (size_t)(input_data.size() * 0.8);

    for (int i = 0; i < train_size; ++i) {
        train_input_data.push_back(input_data[i]);
        train_target_data.push_back(target_data[i]);
    }

    for (int i = train_size; i < input_data.size(); ++i) {
        test_input_data.push_back(input_data[i]);
        test_target_data.push_back(target_data[i]);
    }
}