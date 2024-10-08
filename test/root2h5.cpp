#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <H5Cpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_root> <output_h5> <tree_name>" << std::endl;
        return 1;
    }

    std::string input_root = argv[1];
    std::string output_h5 = argv[2];
    std::string tree_name = argv[3];

    // Open the ROOT file
    TFile* root_file = TFile::Open(input_root.c_str());
    if (!root_file || root_file->IsZombie()) {
        std::cerr << "Error opening ROOT file: " << input_root << std::endl;
        return 1;
    }

    // Get the tree from the ROOT file
    TTree* tree = (TTree*)root_file->Get(tree_name.c_str());
    if (!tree) {
        std::cerr << "Error getting tree: " << tree_name << " from ROOT file." << std::endl;
        return 1;
    }

    // Prepare maps to store the data
    std::map<std::string, std::vector<float>> float_data;
    std::map<std::string, std::vector<int>> int_data;
    std::map<std::string, std::vector<unsigned char>> bool_data;  // Use unsigned char for bool storage
    std::map<std::string, std::vector<std::vector<float>>> vector_float_data;
    std::map<std::string, std::vector<std::vector<int>>> vector_int_data;

    // Loop over all branches in the tree
    TObjArray* branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        std::string branch_name = branch->GetName();
        TLeaf* leaf = branch->GetLeaf(branch_name.c_str());
        std::string branch_type = leaf->GetTypeName();

        if (branch_type == "Float_t") {
            float_data[branch_name] = std::vector<float>();
        } else if (branch_type == "Int_t") {
            int_data[branch_name] = std::vector<int>();
        } else if (branch_type == "Bool_t") {
            bool_data[branch_name] = std::vector<unsigned char>();
        } else if (branch_type == "vector<float>") {
            vector_float_data[branch_name] = std::vector<std::vector<float>>();
        } else if (branch_type == "vector<int>") {
            vector_int_data[branch_name] = std::vector<std::vector<int>>();
        } else {
            std::cerr << "Unsupported branch type: " << branch_type << " for branch: " << branch_name << std::endl;
            continue;
        }
    }

    // Set branch addresses only once before looping over entries
    for (auto& pair : float_data) {
        TBranch* branch = tree->GetBranch(pair.first.c_str());
        float* value = new float;
        branch->SetAddress(value);
        pair.second.push_back(*value);
    }

    for (auto& pair : int_data) {
        TBranch* branch = tree->GetBranch(pair.first.c_str());
        int* value = new int;
        branch->SetAddress(value);
        pair.second.push_back(*value);
    }

    for (auto& pair : bool_data) {
        TBranch* branch = tree->GetBranch(pair.first.c_str());
        bool* value = new bool;
        branch->SetAddress(value);
        pair.second.push_back(static_cast<unsigned char>(*value));
    }

    for (auto& pair : vector_float_data) {
        TBranch* branch = tree->GetBranch(pair.first.c_str());
        std::vector<float>* value = new std::vector<float>;
        branch->SetAddress(&value);
        pair.second.push_back(*value);
    }

    for (auto& pair : vector_int_data) {
        TBranch* branch = tree->GetBranch(pair.first.c_str());
        std::vector<int>* value = new std::vector<int>;
        branch->SetAddress(&value);
        pair.second.push_back(*value);
    }

    // Loop over all entries in the tree and fill the data map
    Long64_t n_entries = tree->GetEntries();
    for (Long64_t entry = 0; entry < n_entries; ++entry) {
        tree->GetEntry(entry);
    }

    // Create an HDF5 file and store the data
    try {
        H5::H5File h5_file(output_h5.c_str(), H5F_ACC_TRUNC);

        for (const auto& pair : float_data) {
            hsize_t dims[1] = { pair.second.size() };
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = h5_file.createDataSet(pair.first, H5::PredType::NATIVE_FLOAT, dataspace);
            dataset.write(pair.second.data(), H5::PredType::NATIVE_FLOAT);
        }

        for (const auto& pair : int_data) {
            hsize_t dims[1] = { pair.second.size() };
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = h5_file.createDataSet(pair.first, H5::PredType::NATIVE_INT, dataspace);
            dataset.write(pair.second.data(), H5::PredType::NATIVE_INT);
        }

        for (const auto& pair : bool_data) {
            hsize_t dims[1] = { pair.second.size() };
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = h5_file.createDataSet(pair.first, H5::PredType::NATIVE_UINT8, dataspace);
            dataset.write(pair.second.data(), H5::PredType::NATIVE_UINT8);  // Store bool as uint8
        }

        for (const auto& pair : vector_float_data) {
            hsize_t dims[2] = { pair.second.size(), pair.second[0].size() };
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = h5_file.createDataSet(pair.first, H5::PredType::NATIVE_FLOAT, dataspace);
            dataset.write(pair.second.data()->data(), H5::PredType::NATIVE_FLOAT);
        }

        for (const auto& pair : vector_int_data) {
            hsize_t dims[2] = { pair.second.size(), pair.second[0].size() };
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = h5_file.createDataSet(pair.first, H5::PredType::NATIVE_INT, dataspace);
            dataset.write(pair.second.data()->data(), H5::PredType::NATIVE_INT);
        }
    } catch (H5::FileIException& e) {
        e.printErrorStack();
        return 1;
    } catch (H5::DataSetIException& e) {
        e.printErrorStack();
        return 1;
    } catch (H5::DataSpaceIException& e) {
        e.printErrorStack();
        return 1;
    }

    std::cout << "Data successfully saved to " << output_h5 << std::endl;
    return 0;
}
