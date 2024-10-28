import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

input_feas = np.load('/home/liang/Workingspace/weaver-core-main/data/embed_data/input_fea.npy')  # [B, c, S]
embed_outputs = np.load('/home/liang/Workingspace/weaver-core-main/data/embed_data/embed_output.npy')  # [S, B, C]

print(f"Original shape of input_feas: {input_feas.shape}")
print(f"Original shape of embed_outputs: {embed_outputs.shape}")

max_samples = 10000

B, c, S = input_feas.shape 
_, B_embed, C_embed = embed_outputs.shape 

input_feas = input_feas[:max_samples]
embed_outputs = embed_outputs[:, :max_samples, :]

input_feas = input_feas.transpose(0, 2, 1).reshape(-1, S * c) 
embed_outputs = embed_outputs.transpose(1, 2, 0).reshape(-1, S * C_embed) 

print(f"Reshaped input_feas: {input_feas.shape}")
print(f"Reshaped embed_outputs: {embed_outputs.shape}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_feas)
scaled_embedded_data = scaler.fit_transform(embed_outputs)

pca = PCA(n_components=0.6)  
principal_components = pca.fit_transform(scaled_embedded_data)

explained_variances = pca.explained_variance_ratio_
print(f"Explained variance ratios for the principal components: {explained_variances}")
print(f"Total number of principal components retained: {len(explained_variances)}")

pc1 = principal_components[:, 0]

correlation_matrix = np.corrcoef(scaled_data.T, pc1.T)[:scaled_data.shape[1], -1]

correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

correlation_matrix = np.abs(correlation_matrix)

physical_observables = ['feature_' + str(i + 1) for i in range(c)] 

mean_correlations = []
for i in range(c):
    mean_correlation = np.mean(correlation_matrix[i * S:(i + 1) * S])
    mean_correlations.append(mean_correlation)

plt.figure(figsize=(12, 6))
sns.barplot(x=physical_observables, y=mean_correlations, palette='coolwarm')
plt.xticks(rotation=90)
plt.xlabel('Physical Observables')
plt.ylabel('Mean Absolute Correlation with PC1')
plt.title('Mean Absolute Correlation between Physical Observables and PC1')
plt.show()

correlation_df = pd.DataFrame({'Physical Observable': physical_observables, 'Mean Absolute Correlation': mean_correlations})
correlation_df.to_csv('test/mean_correlation_with_pc1.csv', index=False)

