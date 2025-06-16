import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

# 1. Extract the data
data_strings = [
    "[0.55540913 0.9888699  0.98988426 0.02365301 0.98839414 0.9447124 ]",
    "[0.9819595  0.99328595 0.0127974  0.98398006 0.95505    0.99138206]",
    "[0.47680253 0.7221648  0.9917835  0.59239656 0.6907198  0.09018178]",
    "[0.99277735 0.44356045 0.9562956  0.9257635  0.49535412 0.00339927]",
    "[0.9922815  0.9953087  0.020839   0.53322375 0.8652929  0.01244537]",
    "[0.348909   0.5528363  0.9783563  0.6552975  0.41247094 0.94134396]",
    "[0.38182127 0.2533229  0.46274093 0.86807114 0.60672253 0.9889193 ]",
    "[0.18809661 0.02296479 0.9760611  0.9476643  0.85538197 0.986331  ]",
    "[0.9908268  0.55724967 0.9765453  0.46605667 0.43917122 0.01384229]",
    "[0.3118141  0.3641659  0.8359696  0.97117764 0.5483904  0.9667982 ]"
]

# Convert strings to a NumPy array
g_vectors = []
for s in data_strings:
    # Remove brackets and split by space, then convert to float
    cleaned_s = s.replace('[', '').replace(']', '')
    g_vectors.append(np.fromstring(cleaned_s, sep=' '))

X = np.array(g_vectors)

# 2. Create and fit the DBSCAN model
# The default min_samples for DBSCAN is 5.
# You might want to adjust min_samples based on your expectations.
# For a small dataset like this, a lower min_samples might be needed to form clusters.
# Let's try with min_samples=2 for demonstration, as min_samples=5 would likely
# result in all points being noise with only 10 data points and eps=0.25.
dbscan = HDBSCAN(min_cluster_size=2, min_samples=3)
clusters = dbscan.fit_predict(X)

# 3. Print the results
print("Data points (g vectors):")
for i, vec in enumerate(X):
    print(f"g{i+1}: {vec}")
print(clusters)

# You can also print which points belong to which cluster
for i, label in enumerate(clusters):
    if label == -1:
        print(f"Point g{i+1} is noise.")
    else:
        print(f"Point g{i+1} belongs to cluster {label}.")