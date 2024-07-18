#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <TFile.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>

void analyze_prediction_results(const std::string& result_path, int& signal_count) {
    TFile* file = TFile::Open(result_path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Failed to open file: " << result_path << std::endl;
        return;
    }

    TTree* tree = nullptr;
    file->GetObject("Events", tree);
    if (!tree) {
        std::cerr << "Failed to get TTree: Events" << std::endl;
        file->Close();
        return;
    }

    std::vector<std::string> branches = {
        "score_label_hh_4b", "score_label_tt_2b4j", "score_label_tth_4b4j",
        "score_label_ttbb_4b4j", "score_label_hbb_4b", "score_label_4b",
        "score_label_2b2j", "score_label_zz_4b", "score_label_zh_4b"
    };

    std::vector<Float_t> scores(branches.size());
    for (size_t i = 0; i < branches.size(); ++i) {
        tree->SetBranchAddress(branches[i].c_str(), &scores[i]);
    }

    Long64_t nentries = tree->GetEntries();
    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);

        auto max_score_iter = std::max_element(scores.begin(), scores.end());
        int max_index = std::distance(scores.begin(), max_score_iter);

        if (branches[max_index] == "score_label_hh_4b") {
            signal_count++;
        }
    }

    file->Close();
}

int main() {
    std::vector<std::string> input_files = {
        "/data/lxiao/Sig_Bkg_EvenT_analysis/sig_hh_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_zz_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_zh_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_tt_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_tth_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_tt2b_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_h2b_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_2b2j_EvenTF",
        "/data/lxiao/Sig_Bkg_EvenT_analysis/bkg_4b_EvenTF"
    };

    std::vector<std::string> output_files = {
        "/home/lxiao/weaver-main/data/sig_hh_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_zz_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_zh_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_tt_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_tth_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_tt2b_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_h2b_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_2b2j_prediction.root",
        "/home/lxiao/weaver-main/data/bkg_4b_prediction.root"
    };

    std::vector<double> scale_factors = {
        2.45e-02, 
        1.28e+00, 
        2.79e-01, 
        1.34e+03, 
        8.45e-01, 
        1.86e+00, 
        1.26e+00, 
        1.48e+04, 
        3.40e+06  
    };

    std::vector<int> signal_counts(9, 0);

    for (size_t i = 0; i < input_files.size(); ++i) {
        std::string command = "bash prediction.sh \"" + input_files[i] + "\" \"" + output_files[i] + "\"";
        int ret = system(command.c_str());
        if (ret != 0) {
            std::cerr << "Error running prediction script for file: " << input_files[i] << std::endl;
            continue;
        }

        analyze_prediction_results(output_files[i], signal_counts[i]);
    }

    double total_signal = 0;
    for (size_t i = 0; i < signal_counts.size(); ++i) {
        total_signal += signal_counts[i] * scale_factors[i];
    }

    std::cout << "Total Signal Count: " << total_signal << std::endl;

    double integrated_luminosity = 3000.0; 
    double sigma_theory = 4.084; 

    double sigma_experiment = total_signal / integrated_luminosity;

    double relative_error = std::abs(sigma_experiment - sigma_theory) / sigma_theory * 100.0;

    std::cout << "Experimental Cross Section: " << sigma_experiment << " fb" << std::endl;
    std::cout << "Theoretical Cross Section: " << sigma_theory << " fb" << std::endl;
    std::cout << "Relative Error: " << relative_error << " %" << std::endl;

    return 0;
}

