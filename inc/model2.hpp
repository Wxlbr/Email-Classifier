#include <vector>
#include <unordered_map>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <algorithm>
#include <sstream>


// Decision Tree Node class
class Node {
public:
    Node* left;
    Node* right;
    int featureIndex;
    int threshold;
    int predictedClass;
    bool isLeaf;

    Node() : left(nullptr), right(nullptr), featureIndex(-1), threshold(0.0),
             predictedClass(-1), isLeaf(false) {}
};

// Decision Tree Classifier class
class DecisionTree {
private:
    Node* root;
    int maxDepth;
    int maxFeatures;

public:
    DecisionTree(int maxDepth, int maxFeatures, Node* root = nullptr) : maxDepth(maxDepth), maxFeatures(maxFeatures), root(root) {}

    // Train decision tree
    void train(const std::vector<std::vector<int>>& features, const std::vector<int>& labels) {
        root = buildTree(features, labels, 0);
    }

    // Predict class label for a single instance
    int predict(const std::vector<int>& instance) {
        Node* current = root;
        while (!current->isLeaf) {
            current = (instance[current->featureIndex] <= current->threshold) ? current->left : current->right;
        }
        return current->predictedClass;
    }

    Node* const getRoot() const {
        return root;
    }

private:

    // Build decision tree recursively
    Node* buildTree(const std::vector<std::vector<int>>& features, const std::vector<int>& labels, int depth) {
        Node* node = new Node();

        if (depth >= maxDepth || labels.empty() || std::equal(labels.begin() + 1, labels.end(), labels.begin())) {
            node->isLeaf = true;
            if (!labels.empty())
                node->predictedClass = getMajorityClass(labels);
            return node;
        }

        int threshold;
        int featureIndex;
        int featureCount = features[0].size();
        std::vector<int> randomFeatureIndices = getRandomIndices(featureCount);

        findBestSplit(features, labels, randomFeatureIndices, featureIndex, threshold);

        node->featureIndex = featureIndex;
        node->threshold = threshold;

        std::vector<std::vector<int>> leftFeatures, rightFeatures;
        std::vector<int> leftLabels, rightLabels;

        // Split data based on threshold
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i][featureIndex] <= threshold) {
                leftFeatures.emplace_back(features[i]);
                leftLabels.emplace_back(labels[i]);
            } else {
                rightFeatures.emplace_back(features[i]);
                rightLabels.emplace_back(labels[i]);
            }
        }

        node->left = buildTree(leftFeatures, leftLabels, depth + 1);
        node->right = buildTree(rightFeatures, rightLabels, depth + 1);
        return node;
    }

    std::vector<int> getRandomIndices(int indicesCount) {

        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<int> features(indicesCount);
        std::iota(features.begin(), features.end(), 0);

        std::shuffle(features.begin(), features.end(), gen);

        std::vector<int> selectedFeatures(features.begin(), features.begin() + maxFeatures);

        return selectedFeatures;
    }

    // Find the best split point based on Gini impurity
    void findBestSplit(const std::vector<std::vector<int>>& instances, const std::vector<int>& labels,
                        const std::vector<int>& randomFeatureIndices, int& bestFeature, int& bestThreshold) {
        int bestGini = std::numeric_limits<int>::max();
        const int numInstances = instances.size();

        for (size_t featureIndex : randomFeatureIndices) {
            int leftCount[2] = {0, 0};
            int rightCount[2] = {numInstances, numInstances}; // Initialize rightCount with total instances
            int lastThreshold;

            for (size_t i = 0; i < numInstances - 1; ++i) {
                int threshold = (instances[i][featureIndex] + instances[i + 1][featureIndex]) / 2.0;

                if (threshold == lastThreshold)
                    continue; // Skip duplicate feature values

                int label = labels[i];
                leftCount[label]++;
                rightCount[label]--;

                // Calculate left and right counts directly
                int leftTotal = leftCount[0] + leftCount[1];
                int rightTotal = rightCount[0] + rightCount[1];

                // Calculate Gini impurity directly
                int giniLeft = calculateGini(leftCount, leftTotal);
                int giniRight = calculateGini(rightCount, rightTotal);

                int gini = (leftTotal / numInstances) * giniLeft + (rightTotal / numInstances) * giniRight;

                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = featureIndex;
                    bestThreshold = threshold;
                }

                lastThreshold = threshold;
            }
        }
    }

    int calculateGini(const int classCounts[2], int numInstances) {
        if (numInstances == 0)
            return 0.0;

        // As the counts are only for binary values, we can calculate the total by adding the two counts
        int classProb0 = classCounts[0] / numInstances;
        int classProb1 = classCounts[1] / numInstances;

        int gini = 1.0 - (classProb0 * classProb0) - (classProb1 * classProb1);
        return gini;
    }

    int getMajorityClass(const std::vector<int>& labels) {
        // Important to execution speed
        int classCounts[2] = {0, 0};

        for (const auto& label : labels)
            classCounts[label]++;

        return (classCounts[0] > classCounts[1]) ? 0 : 1;
    }
};

// Random Forest Classifier class
class RandomForest {
private:
    int numTrees;
    int maxDepth;
    int maxFeatures;
    std::vector<DecisionTree*> trees;

public:
    RandomForest(int numTrees, int maxDepth, int maxFeatures) : numTrees(numTrees), maxDepth(maxDepth), maxFeatures(maxFeatures) {}

    RandomForest(const std::string& filename) : numTrees(10), maxDepth(5), maxFeatures(1) {
        loadModel(filename);
    }

    // Train random forest
    void train(const std::vector<std::vector<int>>& features, const std::vector<int>& labels) {

        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < numTrees; ++i) {
            clock_t startTime = clock();

            DecisionTree* tree = new DecisionTree(maxDepth, maxFeatures);

            tree->train(features, labels);

            trees.push_back(tree);

            clock_t endTime = clock();
            double elapsedTime = double(endTime - startTime) / CLOCKS_PER_SEC * 1000.0;

            std::cout << "Trained tree " << i + 1 << ", " << elapsedTime << " ms" << std::endl;
        }
    }

    int predict(const std::vector<int>& instance) {
        if (trees.empty() || instance.empty())
            return -1;

        int classCounts[2] = {0, 0};
        for (const auto& tree : trees) {
            int prediction = tree->predict(instance);
            classCounts[prediction]++;
        }

        int predictedClass = (classCounts[0] > classCounts[1]) ? 0 : 1;

        return predictedClass;
    }

    int confidence(const std::vector<int>& instance) {
        if (trees.empty() || instance.empty())
            return -1;

        int classCounts[2] = {0, 0};
        for (const auto& tree : trees) {
            int prediction = tree->predict(instance);
            classCounts[prediction]++;
        }

        int maxCount = (classCounts[0] > classCounts[1]) ? classCounts[0] : classCounts[1];
        int confidence = (maxCount / trees.size()) * 100;

        return confidence;
    }

    int calculateAccuracy(const std::vector<std::vector<int>>& testFeatures, const std::vector<int>& testLabels) {
        const size_t featureSize = testFeatures.size();
        int correctPredictions = 0;

        for (size_t i = 0; i < featureSize; ++i) {
            if (predict(testFeatures[i]) == testLabels[i]) {
                correctPredictions++;
            }
        }

        return correctPredictions / featureSize;
    }

    void saveModel(const std::string& filename) {
        std::stringstream stream;

        // Write the model properties
        stream.write(reinterpret_cast<const char*>(&numTrees), sizeof(numTrees));
        stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(maxDepth));
        stream.write(reinterpret_cast<const char*>(&maxFeatures), sizeof(maxFeatures));

        // Write each decision tree
        for (const auto& tree : trees) {
            saveDecisionTreeToStream(stream, tree->getRoot());
        }

        std::ofstream file(filename, std::ios::binary);
        if (file) {
            std::string serialisedData = stream.str();
            file.write(serialisedData.c_str(), serialisedData.size());
            file.close();
        } else {
            std::cerr << "Failed to open file" << std::endl;
        }
    }

    void loadModel(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open the file" << std::endl;
            return;
        }

        std::stringstream stream;
        stream << file.rdbuf();

        // Read the model properties
        stream.read(reinterpret_cast<char*>(&numTrees), sizeof(numTrees));
        stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(maxDepth));
        stream.read(reinterpret_cast<char*>(&maxFeatures), sizeof(maxFeatures));

        // Load each decision tree
        trees.clear();
        for (int i = 0; i < numTrees; ++i) {
            Node* root = loadDecisionTreeFromStream(stream);
            DecisionTree* tree = new DecisionTree(maxDepth, maxFeatures, root);
            trees.push_back(tree);
        }

        file.close();
    }

private:
    // Helper function to save a decision tree recursively
    void saveDecisionTreeToStream(std::ostream& stream, Node* node) {
        if (!node) {
            return;
        }

        // Save node properties
        stream.write(reinterpret_cast<const char*>(&node->featureIndex), sizeof(node->featureIndex));
        stream.write(reinterpret_cast<const char*>(&node->threshold), sizeof(node->threshold));
        stream.write(reinterpret_cast<const char*>(&node->predictedClass), sizeof(node->predictedClass));
        stream.write(reinterpret_cast<const char*>(&node->isLeaf), sizeof(node->isLeaf));

        // Recursively save left and right subtrees
        saveDecisionTreeToStream(stream, node->left);
        saveDecisionTreeToStream(stream, node->right);
    }

    // Helper function to load a decision tree recursively
    Node* loadDecisionTreeFromStream(std::istream& stream) {
        Node* node = new Node();
        // Load node properties
        stream.read(reinterpret_cast<char*>(&node->featureIndex), sizeof(node->featureIndex));
        stream.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
        stream.read(reinterpret_cast<char*>(&node->predictedClass), sizeof(node->predictedClass));
        stream.read(reinterpret_cast<char*>(&node->isLeaf), sizeof(node->isLeaf));

        // Recursively load left and right subtrees if they exist
        if (!node->isLeaf) {
            node->left = loadDecisionTreeFromStream(stream);
            node->right = loadDecisionTreeFromStream(stream);
        }

        return node;
    }
};

// Split data into train and test sets
void trainTestSplit(const std::vector<std::vector<int>>& allFeatures, const std::vector<int>& allLabels,
                    std::vector<std::vector<int>>& trainFeatures, std::vector<int>& trainLabels,
                    std::vector<std::vector<int>>& testFeatures, std::vector<int>& testLabels,
                    int trainRatio = 0.8) {
    const size_t numFeatures = allFeatures.size();

    if (numFeatures != allLabels.size()) {
        std::cerr << "Mismatch between features and labels!" << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    int numTrainInstances = std::round(numFeatures * trainRatio);

    std::vector<int> indices(numFeatures);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < numFeatures; ++i) {
        if (i < numTrainInstances) {
            trainFeatures.emplace_back(allFeatures[indices[i]]);
            trainLabels.emplace_back(allLabels[indices[i]]);
        } else {
            testFeatures.emplace_back(allFeatures[indices[i]]);
            testLabels.emplace_back(allLabels[indices[i]]);
        }
    }
}
