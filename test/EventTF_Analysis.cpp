#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

int getMaxScoreIndex(const std::vector<float>& scores) {
    int maxIndex = 0;
    float maxValue = scores[0];
    for (int i = 1; i < scores.size(); ++i) {
        if (scores[i] > maxValue) {
            maxValue = scores[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_root_path> <output_root_path>" << std::endl;
        return 1;
    }

    std::string input_root_path = argv[1];
    std::string output_root_path = argv[2];

    std::string command = "./prediction.sh " + input_root_path + " " + output_root_path;
    if (system(command.c_str()) != 0) {
        std::cerr << "Error: Failed to execute prediction script." << std::endl;
        return 1;
    }

    TFile* input_file = TFile::Open(input_root_path.c_str());
    TTree* input_tree = (TTree*)input_file->Get("tree");

    TFile* prediction_file = TFile::Open(output_root_path.c_str());
    TTree* prediction_tree = (TTree*)prediction_file->Get("Events");

    std::vector<int> bins = {0, 372.07, 413.88, 463.20, 544.72, 14000};
    std::vector<int> signal_counts(bins.size() - 1, 0);
    std::vector<int> background_counts(bins.size() - 1, 0);

    float sqrt_s;
    float score_label_hh_4b, score_label_tt_2b4j, score_label_tth_4b4j, score_label_ttbb_4b4j;
    float score_label_hbb_4b, score_label_4b, score_label_2b2j, score_label_zz_4b, score_label_zh_4b;

    input_tree->SetBranchAddress("sqrt(s)", &sqrt_s);
    prediction_tree->SetBranchAddress("score_label_hh_4b", &score_label_hh_4b);
    prediction_tree->SetBranchAddress("score_label_tt_2b4j", &score_label_tt_2b4j);
    prediction_tree->SetBranchAddress("score_label_tth_4b4j", &score_label_tth_4b4j);
    prediction_tree->SetBranchAddress("score_label_ttbb_4b4j", &score_label_ttbb_4b4j);
    prediction_tree->SetBranchAddress("score_label_hbb_4b", &score_label_hbb_4b);
    prediction_tree->SetBranchAddress("score_label_4b", &score_label_4b);
    prediction_tree->SetBranchAddress("score_label_2b2j", &score_label_2b2j);
    prediction_tree->SetBranchAddress("score_label_zz_4b", &score_label_zz_4b);
    prediction_tree->SetBranchAddress("score_label_zh_4b", &score_label_zh_4b);

    int n_entries = input_tree->GetEntries();
    for (int i = 0; i < n_entries; ++i) {
        input_tree->GetEntry(i);
        prediction_tree->GetEntry(i);

        std::vector<float> scores = {
            score_label_hh_4b, score_label_tt_2b4j, score_label_tth_4b4j, score_label_ttbb_4b4j,
            score_label_hbb_4b, score_label_4b, score_label_2b2j, score_label_zz_4b, score_label_zh_4b
        };

        int max_score_index = getMaxScoreIndex(scores);
        bool is_signal = (max_score_index == 0);

        for (int bin_idx = 0; bin_idx < bins.size() - 1; ++bin_idx) {
            if (bins[bin_idx] <= sqrt_s && sqrt_s < bins[bin_idx + 1]) {
                if (is_signal) {
                    signal_counts[bin_idx]++;
                } else {
                    background_counts[bin_idx]++;
                }
                break;
            }
        }
    }

    for (int bin_idx = 0; bin_idx < bins.size() - 1; ++bin_idx) {
        std::cout << "Bin [" << bins[bin_idx] << ", " << bins[bin_idx + 1] 
                  << "): Signal = " << signal_counts[bin_idx] 
                  << ", Background = " << background_counts[bin_idx] << std::endl;
    }

    input_file->Close();
    prediction_file->Close();

    return 0;
}

