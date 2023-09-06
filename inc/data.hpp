#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>

struct CSVRecord {
    std::string text;
    int label;
};

// Function to tokenize a string into words
std::vector<std::string> tokenise(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (std::isalpha(c)) {
            token += std::tolower(c);
        } else {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to calculate the TF-IDF of a word in a document
double tfidf(const std::string& word, const std::vector<std::string>& document, const std::map<std::string, std::vector<std::vector<std::string>>>& documents) {
    int termFrequency = 0;
    for (const std::string& w : document) {
        if (w == word) {
            termFrequency++;
        }
    }

    double inverseDocumentFrequency = 0.0;
    if (documents.find(word) != documents.end()) {
        inverseDocumentFrequency = log(documents.size() / (double)documents.at(word).size());
    }

    return termFrequency * inverseDocumentFrequency;
}

// Function to create a map of words to documents
std::map<std::string, std::vector<std::vector<std::string>>> createDocuments(const std::vector<CSVRecord>& data) {
    std::map<std::string, std::vector<std::vector<std::string>>> documents;
    for (const CSVRecord& record : data) {
        std::vector<std::string> tokens = tokenise(record.text);
        for (const std::string& token : tokens) {
            if (documents.find(token) == documents.end()) {
                documents[token] = std::vector<std::vector<std::string>>();
            }
            documents[token].push_back(tokens);
        }
    }
    return documents;
}

std::string cleanText(const std::string& text) {
    std::string cleanedText = text;

    // Remove all punctuation
    for (int i = 0; i < cleanedText.size(); i++) {
        if (ispunct(cleanedText[i])) {
            cleanedText.erase(i--, 1);
        }
    }

    // Convert to lowercase
    for (int i = 0; i < cleanedText.size(); i++) {
        cleanedText[i] = tolower(cleanedText[i]);
    }

    // Convert all whitespace to single spaces
    for (int i = 0; i < cleanedText.size(); i++) {
        if (isspace(cleanedText[i])) {
            while (isspace(cleanedText[i + 1])) {
                cleanedText.erase(i + 1, 1);
            }
        }
    }

    // Remove web links
    while (cleanedText.find("http") != std::string::npos) {
        int start = cleanedText.find("http");
        int end = cleanedText.find(" ", start);
        cleanedText.erase(start, end - start);
    }

    // Remove all numbers
    for (int i = 0; i < cleanedText.size(); i++) {
        if (isdigit(cleanedText[i])) {
            cleanedText.erase(i--, 1);
        }
    }

    // Remove first occourance of "subject" and "re"
    if (cleanedText.find("subject") != std::string::npos) {
        cleanedText.erase(cleanedText.find("subject"), 7);
    }

    if (cleanedText.find("re") != std::string::npos) {
        cleanedText.erase(cleanedText.find("re"), 2);
    }

    return cleanedText;
};

// Load CSV file
bool loadCSV(const std::string& filename, std::vector<CSVRecord>& data, std::string textHeader, std::string flagHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Skip the first line (header)
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;

    int textIndex = -1;
    int flagIndex = -1;
    int i = 0;

    while (std::getline(ss, cell, ',')) {
        if (cell == textHeader) {
            textIndex = i;
        } else if (cell == flagHeader) {
            flagIndex = i;
        }

        i++;
    }

    if (textIndex == -1) {
        std::cerr << "Text header not found: " << textHeader << std::endl;
        return false;
    }

    if (flagIndex == -1) {
        std::cerr << "Flag header not found: " << flagHeader << std::endl;
        return false;
    }

    std::stringstream dataStream;

    // Read the remaining data lines into a single stringstream
    while (std::getline(file, line)) {
        dataStream << line << " ,";
    }

    // Replace all occourances of "" with ''
    std::string dataString = dataStream.str();

    while (dataString.find("\"\"") != std::string::npos) {
        dataString.replace(dataString.find("\"\""), 2, "\'\'");
    }

    dataStream = std::stringstream(dataString);

    std::string token;
    int tokenIndex = 0;

    std::string text;
    int label;

    // Parse the data from the stringstream
    while (std::getline(dataStream, token, ',')) {
        if (token.front() == '"') {
            std::string temp = token;
            while (temp.back() != '"' && std::getline(dataStream, token, ',')) {
                temp += " ," + token;
            }
            text = cleanText(temp);
            // std::cout << text << std::endl;
        }

        if (tokenIndex % 4 == flagIndex) {
            // std::cout << token << std::endl;
            label = std::stoi(token);
        }

        // std::cout << tokenIndex << std::endl;

        tokenIndex += 1;

        if (label != -1) {
            data.push_back(CSVRecord{text, label});
            text = "";
            label = -1;
        }
    }

    file.close();
    return true;
}