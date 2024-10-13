#include <iostream>
#include <vector>
#include <algorithm>
#include "TFile.h"
#include "TTree.h"
#include <cmath>
#include <numeric>

// Function to find the optimal threshold
double findOptimalThreshold(const char* rootFilePath, const char* treeName = "Events") {
    // Open ROOT file
    TFile* file = TFile::Open(rootFilePath);
    if (!file || file->IsZombie()) {
        std::cerr << "Unable to open ROOT file: " << rootFilePath << std::endl;
        return -1;
    }

    // Get tree from the file
    TTree* tree = (TTree*)file->Get(treeName);
    if (!tree) {
        std::cerr << "Tree not found: " << treeName << std::endl;
        file->Close();
        return -1;
    }

    // Prepare variables
    float score_hh_4b, score_tt_2b2j, score_tth_4b4j, score_ttbb_4b4j, score_hbb_4b, score_4b, score_2b2j, score_zz_4b, score_zh_4b;
    bool label_hh_4b, label_tt_2b2j, label_tth_4b4j, label_ttbb_4b4j, label_hbb_4b, label_4b, label_2b2j, label_zz_4b, label_zh_4b;

    tree->SetBranchAddress("score_label_hh_4b", &score_hh_4b);
    tree->SetBranchAddress("score_label_tt_2b4j", &score_tt_2b2j);
    tree->SetBranchAddress("score_label_tth_4b4j", &score_tth_4b4j);
    tree->SetBranchAddress("score_label_ttbb_4b4j", &score_ttbb_4b4j);
    tree->SetBranchAddress("score_label_hbb_4b", &score_hbb_4b);
    tree->SetBranchAddress("score_label_4b", &score_4b);
    tree->SetBranchAddress("score_label_2b2j", &score_2b2j);
    tree->SetBranchAddress("score_label_zz_4b", &score_zz_4b);
    tree->SetBranchAddress("score_label_zh_4b", &score_zh_4b);

    tree->SetBranchAddress("label_hh_4b", &label_hh_4b);
    tree->SetBranchAddress("label_tt_2b4j", &label_tt_2b2j);
    tree->SetBranchAddress("label_tth_4b4j", &label_tth_4b4j);
    tree->SetBranchAddress("label_ttbb_4b4j", &label_ttbb_4b4j);
    tree->SetBranchAddress("label_hbb_4b", &label_hbb_4b);
    tree->SetBranchAddress("label_4b", &label_4b);
    tree->SetBranchAddress("label_2b2j", &label_2b2j);
    tree->SetBranchAddress("label_zz_4b", &label_zz_4b);
    tree->SetBranchAddress("label_zh_4b", &label_zh_4b);

    double bestThreshold = 0.0;
    double minRelativeError = std::numeric_limits<double>::max();

    const int numLabels = 9;
    std::vector<double> relativeErrors;

    // Scan different thresholds
    for (double threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
        std::vector<int> totalLabelCount(numLabels, 0);
        std::vector<int> predictedAsHH4BCount(numLabels, 0);
        Long64_t nEntries = tree->GetEntries();

        // Loop over all entries to collect predictions
        for (Long64_t i = 0; i < nEntries; ++i) {
            tree->GetEntry(i);

            std::vector<float> scores = {score_tt_2b2j, score_tth_4b4j, score_ttbb_4b4j, score_hbb_4b, score_4b, score_2b2j, score_zz_4b, score_zh_4b};
            std::vector<bool> labels = {label_tt_2b2j, label_tth_4b4j, label_ttbb_4b4j, label_hbb_4b, label_4b, label_2b2j, label_zz_4b, label_zh_4b};

            int trueLabelIndex = 0;
            if (label_hh_4b) trueLabelIndex = 0;
            else {
                for (int j = 0; j < 8; ++j) {
                    if (labels[j]) {
                        trueLabelIndex = j + 1;
                        break;
                    }
                }
            }

            totalLabelCount[trueLabelIndex] += 1;

            if (score_hh_4b > threshold) {
                predictedAsHH4BCount[trueLabelIndex] += 1;
            }
        }

        // Calculate the probability of being predicted as hh_4b
        std::vector<double> predictionProbabilities(numLabels, 0.0);
        for (int i = 0; i < numLabels; ++i) {
            if (totalLabelCount[i] > 0) {
                predictionProbabilities[i] = static_cast<double>(predictedAsHH4BCount[i]) / totalLabelCount[i];
            }
        }

        // Check if the prediction accuracy of label_hh_4b is less than 50%
        if (predictionProbabilities[0] < 0.50) {
            // If the accuracy is less than 50%, skip this threshold
            continue;
        }

        // Constants
        double sigma_sm_di_higgs = 1.454e+01;
        double BR_hh_4b = 0.53 * 0.53;
        double L = 3000;
        std::vector<double> sigma_b = {2.237e+05, 9.370e+01, 5.996e+03, 2.506e+01, 2.475e+06, 5.67e+08, 2.135e+02, 4.647e+01};

        // Extract ps and pb
        double ps = predictionProbabilities[0]; // Signal
        std::vector<double> pb(predictionProbabilities.begin() + 1, predictionProbabilities.end()); // Background

        // Numerator
        double numerator = ps * sigma_sm_di_higgs * BR_hh_4b * L + std::inner_product(pb.begin(), pb.end(), sigma_b.begin(), 0.0) * L;

        // Denominator
        double denominator = ps * BR_hh_4b * L * sigma_sm_di_higgs;

        // Calculate relative error
        double relativeError = std::sqrt(3.841 * numerator) / denominator;

        relativeErrors.push_back(relativeError);

        if (relativeError < minRelativeError) {
            minRelativeError = relativeError;
            bestThreshold = threshold;
        }
    }

    // Output the result
    std::cout << "Optimal threshold is: " << bestThreshold << ", with a minimum relative error of: " << minRelativeError << std::endl;

    // Close the file
    file->Close();

    return bestThreshold;
}

int main(int argc, char** argv) {
    // Check if the ROOT file path is provided
    if (argc < 2) {
        std::cerr << "Please provide the ROOT file path!" << std::endl;
        return 1;
    }

    // Get the ROOT file path from the command line
    const char* rootFilePath = argv[1];
    
    findOptimalThreshold(rootFilePath);
    return 0;
}

