###########Python code for PCA analysis of matrix data

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Your distance matrix as a nested list
distance_matrix = [
    [0, 0.00341, 0.0022, 0.00252, 0.00225, 0.0028, 0.00219, 0.00214, 0.00314, 0.00293, 0.00261, 0.00247, 0.00275, 0.00128, 0.00325],
    [0.00341, 0, 0.00374, 0.00407, 0.00369, 0.0044, 0.0035, 0.00366, 0.0047, 0.00421, 0.00418, 0.00378, 0.00392, 0.00284, 0.00457],
    [0.0022, 0.00374, 0, 0.00267, 0.00226, 0.00307, 0.0022, 0.00215, 0.00317, 0.00292, 0.00263, 0.00253, 0.00292, 0.00127, 0.00322],
    [0.00252, 0.00407, 0.00267, 0, 0.00269, 0.00254, 0.00265, 0.00258, 0.0036, 0.00305, 0.00305, 0.00294, 0.00333, 0.00172, 0.00362],
    [0.00225, 0.00369, 0.00226, 0.00269, 0, 0.00285, 0.0019, 0.00165, 0.00318, 0.00281, 0.00263, 0.00206, 0.00293, 0.0013, 0.00315],
    [0.0028, 0.0044, 0.00307, 0.00254, 0.00285, 0, 0.00251, 0.00286, 0.00402, 0.00347, 0.00346, 0.00316, 0.00369, 0.00213, 0.00393],
    [0.00219, 0.0035, 0.0022, 0.00265, 0.0019, 0.00251, 0, 0.00195, 0.00311, 0.00249, 0.00259, 0.0023, 0.00287, 0.00125, 0.00251],
    [0.00214, 0.00366, 0.00215, 0.00258, 0.00165, 0.00286, 0.00195, 0, 0.00307, 0.00275, 0.00252, 0.00193, 0.00284, 0.00119, 0.00312],
    [0.00314, 0.0047, 0.00317, 0.0036, 0.00318, 0.00402, 0.00311, 0.00307, 0, 0.00387, 0.00338, 0.00347, 0.00388, 0.00221, 0.00418],
    [0.00293, 0.00421, 0.00292, 0.00305, 0.00281, 0.00347, 0.00249, 0.00275, 0.00387, 0, 0.00333, 0.00312, 0.00356, 0.00199, 0.00312],
    [0.00261, 0.00418, 0.00263, 0.00305, 0.00263, 0.00346, 0.00259, 0.00252, 0.00338, 0.00333, 0, 0.00292, 0.0033, 0.00166, 0.00366],
    [0.00247, 0.00378, 0.00253, 0.00294, 0.00206, 0.00316, 0.0023, 0.00193, 0.00347, 0.00312, 0.00292, 0, 0.00317, 0.00159, 0.00347],
    [0.00275, 0.00392, 0.00292, 0.00333, 0.00293, 0.00369, 0.00287, 0.00284, 0.00388, 0.00356, 0.0033, 0.00317, 0, 0.00199, 0.00388],
    [0.00128, 0.00284, 0.00127, 0.00172, 0.0013, 0.00213, 0.00125, 0.00119, 0.00221, 0.00199, 0.00166, 0.00159, 0.00199, 0, 0.00232],
    [0.00325, 0.00457, 0.00322, 0.00362, 0.00315, 0.00393, 0.00251, 0.00312, 0.00418, 0.00312, 0.00366, 0.00347, 0.00388, 0.00232, 0]
]

# List of countries corresponding to the distance matrix
countries = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

# Convert the distance matrix to a pandas DataFrame
distance_df = pd.DataFrame(distance_matrix, index=countries, columns=countries)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(distance_df)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Country'] = countries

# Plot the PCA results
plt.figure(figsize=(10, 8))
for country in pca_df['Country'].unique():
    subset = pca_df[pca_df['Country'] == country]
    plt.scatter(subset['PC1'], subset['PC2'], label=country)

# Add the explained variance to the plot
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% Variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% Variance)')
plt.title('PCA of ToBRFV Genetic Distances')
plt.legend()
plt.show()
