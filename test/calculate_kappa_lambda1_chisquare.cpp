#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <TFile.h>
#include <TTree.h>
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

    // 定义所有可能的 score branches
    std::vector<std::string> branches = {
        "score_label_hh_4b", "score_label_tt_2b4j", "score_label_tth_4b4j",
        "score_label_ttbb_4b4j", "score_label_hbb_4b", "score_label_4b",
        "score_label_2b2j", "score_label_zz_4b", "score_label_zh_4b"
    };

    // 为每个 branch 创建一个变量以存储值
    std::vector<Float_t> scores(branches.size());
    for (size_t i = 0; i < branches.size(); ++i) {
        tree->SetBranchAddress(branches[i].c_str(), &scores[i]);
    }

    // 遍历每一个 entry
    Long64_t nentries = tree->GetEntries();
    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);

        // 找出当前 entry 中最大的 score
        auto max_score_iter = std::max_element(scores.begin(), scores.end());
        int max_index = std::distance(scores.begin(), max_score_iter);

        // 如果最大值是 score_label_hh_4b 则为信号
        if (branches[max_index] == "score_label_hh_4b") {
            signal_count++;
        }
    }

    file->Close();
}

int main() {
    // 固定输入路径
    std::vector<std::string> input_files = {
        "/data/lxiao/Sig_Bkg_EvenT/test/bkg_2b2j_EvenTF/*.root",
        //"/path/to/input/background1.root",
        //"/path/to/input/background2.root",
        //"/path/to/input/background3.root",
        //"/path/to/input/background4.root",
        //"/path/to/input/background5.root",
        //"/path/to/input/background6.root",
        //"/path/to/input/background7.root",
        //"/path/to/input/background8.root"
    };

    // 固定输出路径
    std::vector<std::string> output_files = {
        "/home/lxiao/weaver-main/data/2b2j_prediction.root",
        //"/path/to/output/background1.root",
        //"/path/to/output/background2.root",
        //"/path/to/output/background3.root",
        //"/path/to/output/background4.root",
        //"/path/to/output/background5.root",
        //"/path/to/output/background6.root",
        //"/path/to/output/background7.root",
        //"/path/to/output/background8.root"
    };

    // 固定S_sm的输入和输出路径
    std::string sm_input_file = "/data/lxiao/Sig_Bkg_EvenT/test/sig_hh_EvenTF/*.root";
    std::string sm_output_file = "/home/lxiao/weaver-main/data/hh_prediction.root";

    // 缩放因子
    std::vector<double> scale_factors = {
        2.0, // 例如：信号的缩放因子
        //1.0, // 例如：背景1的缩放因子
        //1.0, // 例如：背景2的缩放因子
        //1.0, // 例如：背景3的缩放因子
        //1.0, // 例如：背景4的缩放因子
        //1.0, // 例如：背景5的缩放因子
        //1.0, // 例如：背景6的缩放因子
        //1.0, // 例如：背景7的缩放因子
        //1.0  // 例如：背景8的缩放因子
    };

    std::vector<int> signal_counts(9, 0);
    int sm_signal_count = 0;

    // 调用预测脚本并分析结果
    for (size_t i = 0; i < input_files.size(); ++i) {
        std::string command = "bash prediction.sh \"" + input_files[i] + "\" \"" + output_files[i] + "\"";
        int ret = system(command.c_str());
        if (ret != 0) {
            std::cerr << "Error running prediction script for file: " << input_files[i] << std::endl;
            continue;
        }

        analyze_prediction_results(output_files[i], signal_counts[i]);
    }

    // 分析S_sm的结果
    std::string sm_command = "bash prediction.sh \"" + sm_input_file + "\" \"" + sm_output_file + "\"";
    int sm_ret = system(sm_command.c_str());
    if (sm_ret != 0) {
        std::cerr << "Error running prediction script for file: " << sm_input_file << std::endl;
    } else {
        analyze_prediction_results(sm_output_file, sm_signal_count);
    }

    // 计算总的信号数量
    double total_signal = 0;
    for (size_t i = 0; i < signal_counts.size(); ++i) {
        total_signal += signal_counts[i] * scale_factors[i];
    }

    std::cout << "Total Signal Count (S): " << total_signal << std::endl;
    std::cout << "SM Signal Count (S_sm): " << sm_signal_count << std::endl;

    // 计算chi-square
    double chi_square = (total_signal - sm_signal_count) * (total_signal - sm_signal_count) /
                        (sm_signal_count + (0.003 * sm_signal_count) * (0.003 * sm_signal_count));

    std::cout << "Chi-square: " << chi_square << std::endl;

    return 0;
}

