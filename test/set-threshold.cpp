#include <iostream>
#include <vector>
#include <algorithm>
#include "TFile.h"
#include "TTree.h"

double findOptimalThreshold(const char* rootFilePath, const char* treeName = "Events") {
    // Open ROOT file
    TFile* file = TFile::Open(rootFilePath);
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open ROOT file: " << rootFilePath << std::endl;
        return -1;
    }
    
    // Get tree from file
    TTree* tree = (TTree*)file->Get(treeName);
    if (!tree) {
        std::cerr << "Cannot find tree: " << treeName << std::endl;
        file->Close();
        return -1;
    }

    // Prepare variables
    float score_hh_4b, score_tt_2b2j, score_tth_4b4j, score_ttbb_4b4j, score_hbb_4b, score_4b, score_2b2j, score_zz_4b, score_zh_4b;
    bool label_hh_4b, label_tt_2b2j, label_tth_4b4j, label_ttbb_4b4j, label_hbb_4b, label_4b, label_2b2j, label_zz_4b, label_zh_4b;
    
    tree->SetBranchAddress("score_label_hh_4b", &score_hh_4b);
    tree->SetBranchAddress("score_label_tt_2b2j", &score_tt_2b2j);
    tree->SetBranchAddress("score_label_tth_4b4j", &score_tth_4b4j);
    tree->SetBranchAddress("score_label_ttbb_4b4j", &score_ttbb_4b4j);
    tree->SetBranchAddress("score_label_hbb_4b", &score_hbb_4b);
    tree->SetBranchAddress("score_label_4b", &score_4b);
    tree->SetBranchAddress("score_label_2b2j", &score_2b2j);
    tree->SetBranchAddress("score_label_zz_4b", &score_zz_4b);
    tree->SetBranchAddress("score_label_zh_4b", &score_zh_4b);
    
    tree->SetBranchAddress("label_hh_4b", &label_hh_4b);
    tree->SetBranchAddress("label_tt_2b2j", &label_tt_2b2j);
    tree->SetBranchAddress("label_tth_4b4j", &label_tth_4b4j);
    tree->SetBranchAddress("label_ttbb_4b4j", &label_ttbb_4b4j);
    tree->SetBranchAddress("label_hbb_4b", &label_hbb_4b);
    tree->SetBranchAddress("label_4b", &label_4b);
    tree->SetBranchAddress("label_2b2j", &label_2b2j);
    tree->SetBranchAddress("label_zz_4b", &label_zz_4b);
    tree->SetBranchAddress("label_zh_4b", &label_zh_4b);

    double bestThreshold = 0.0;
    double minErrorSum;
    bool firstIteration = true;

    // Scan thresholds
    for (double threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
        int errorSum = 0;
        Long64_t nEntries = tree->GetEntries();

        for (Long64_t i = 0; i < nEntries; ++i) {
            tree->GetEntry(i);

            std::vector<float> scores = {score_tt_2b2j, score_tth_4b4j, score_ttbb_4b4j, score_hbb_4b, score_4b, score_2b2j, score_zz_4b, score_zh_4b};
            std::vector<bool> labels = {label_tt_2b2j, label_tth_4b4j, label_ttbb_4b4j, label_hbb_4b, label_4b, label_2b2j, label_zz_4b, label_zh_4b};

            int predictedLabel = 0;
            if (score_hh_4b > threshold) {
                predictedLabel = 0;
            } else {
                // Find the highest scoring label among the remaining 8 labels
                auto maxElementIter = std::max_element(scores.begin(), scores.end());
                predictedLabel = std::distance(scores.begin(), maxElementIter) + 1;
            }

            // Calculate error
            if (predictedLabel == 0 && !label_hh_4b) {
                errorSum += 1;
            }
        }

        double errorRate = static_cast<double>(errorSum) / nEntries;

        // Initialize minErrorSum on the first iteration
        if (firstIteration) {
            minErrorSum = errorRate;
            bestThreshold = threshold;
            firstIteration = false;
        } else if (errorRate < minErrorSum) {
            minErrorSum = errorRate;
            bestThreshold = threshold;
        }
    }

    // Output the result
    std::cout << "Optimal threshold is: " << bestThreshold << ", with a minimum error rate of: " << minErrorSum << std::endl;

    // Close the file
    file->Close();

    return bestThreshold;
}

int main() {
    const char* rootFilePath = ".root";
    findOptimalThreshold(rootFilePath);
    return 0;
}

