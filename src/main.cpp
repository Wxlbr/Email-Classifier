#include <iostream>

#include "data.hpp"

int main() {

    std::cout << "Program is running..." << std::endl;

    std::vector<std::string> header = {"text", "label"};
    std::vector<CSVRecord> data;
    std::string textHeader = "text";
    std::string flagHeader = "label_num";

    loadCSV("../inc/kaggleDataset.csv", data, textHeader, flagHeader);

    std::cout << "Data size: " << data.size() << std::endl;

    for (int i = 0; i < 5; i++) {
        std::cout << data[i].text << std::endl;
        std::cout << data[i].label << std::endl;
    }


    return 0;
}