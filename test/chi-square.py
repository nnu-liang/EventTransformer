import uproot
import numpy as np
from scipy.stats import chisquare, chi2

file_paths = [
    ("/home/liang/Workingspace/weaver-core-main/data/plot_feature_distribution/Bkg_2B2J_32_204524821716_EvenTF.root", "2b2j"),
    ("/home/liang/Workingspace/weaver-core-main/data/plot_feature_distribution/Bkg_4B_32_204524821377_EvenTF.root", "4b")
]

branches = [
    "event_px", "event_energy", "event_nparticles", 
    "jet_pt", "jet_px", "jet_py", "jet_pz", "jet_energy", "jet_ncharged", "jet_nneutral", 
    "part_pt", "part_energy", "part_charge", 
    "jet_dEta_two_jets", "jet_dPhi_two_jets", 
    "part_ptrel_particle_jet", "part_erel_particle_jet", "part_dEta_particle_jet", "part_dPhi_particle_jet",
    "jet_sdmass", "jet_neutralenergyfraction", "jet_chargedenergyfraction", "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4"
]

def process_branch(branch):
    trees = [(uproot.open(file_path)["tree"], label) for file_path, label in file_paths]

    data_list = []

    for (tree, label) in trees:
        data = tree[branch].array(library="np")
        if data.dtype == object:
            data = np.concatenate(data).astype(np.float64)
        else:
            data = data.astype(np.float64)
        print(f"Branch: {branch}, Label: {label}, Data shape: {data.shape}, Data type: {data.dtype}")
        data_list.append(data)

    if not all(isinstance(d, np.ndarray) and d.ndim == 1 for d in data_list):
        raise ValueError("All data arrays must be one-dimensional.")

    min_val1 = data_list[0].min()
    min_val2 = data_list[1].min()
    max_val1 = data_list[0].max()
    max_val2 = data_list[1].max()
    print(f"Min values: {min_val1}, {min_val2}")
    print(f"Max values: {max_val1}, {max_val2}")

    min_val = min(min_val1, min_val2)
    max_val = max(max_val1, max_val2)

    n_bins = 50

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    counts1, bin_edges = np.histogram(data_list[0], bins=bin_edges)
    counts2, _ = np.histogram(data_list[1], bins=bin_edges)

    def merge_bins(counts1, counts2, bin_edges, min_expected_count=5):
        new_counts1 = []
        new_counts2 = []
        new_bin_edges = [bin_edges[0]]
        i = 0

        while i < len(counts1):
            if counts2[i] < min_expected_count:
                if i > 0:
                    new_counts1[-1] += counts1[i]
                    new_counts2[-1] += counts2[i]
                else:
                    new_counts1.append(counts1[i] + counts1[i + 1])
                    new_counts2.append(counts2[i] + counts2[i + 1])
                    new_bin_edges.append(bin_edges[i + 2])
                    i += 1
            else:
                new_counts1.append(counts1[i])
                new_counts2.append(counts2[i])
                new_bin_edges.append(bin_edges[i + 1])
            i += 1

        return np.array(new_counts1), np.array(new_counts2), np.array(new_bin_edges)

    while True:
        new_counts1, new_counts2, new_bin_edges = merge_bins(counts1, counts2, bin_edges)
        if np.any(new_counts2 < 5):
            counts1, counts2, bin_edges = new_counts1, new_counts2, new_bin_edges
        else:
            break

    print(f"Branch: {branch}")
    print(f"New Counts1: {new_counts1}")
    print(f"New Counts2: {new_counts2}")
    print(f"New Bin Edges: {new_bin_edges}")

    total_counts1 = np.sum(new_counts1)
    total_counts2 = np.sum(new_counts2)
    scale_factor = total_counts1 / total_counts2
    new_counts2_scaled = new_counts2 * scale_factor

    print(f"Total Counts1: {total_counts1}")
    print(f"Total Counts2 (Scaled): {np.sum(new_counts2_scaled)}")

    chi2_stat, p_val = chisquare(f_obs=new_counts1, f_exp=new_counts2_scaled)

    print(f"Chi-Square Test between 2b2j and 4b for {branch}: Chi2 statistic = {chi2_stat}, p-value = {p_val}")

    alpha = 0.05
    df = len(new_counts1) - 1

    critical_value = chi2.ppf(1 - alpha, df)
    print(f"Critical value for alpha = {alpha} and df = {df}: {critical_value}")

    if chi2_stat > critical_value:
        print("Reject the null hypothesis: The distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not significantly different.")
    print("\n")

for branch in branches:
    process_branch(branch)

