import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os  # For file existence check

# Load sample customer dataset
df = pd.read_excel('sample_customer_data.xlsx')  # You can create or use mock data with columns like Age, Income, SpendingScore

# Basic EDA
print(df.head())
print(df.describe())

# Select relevant features
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Segment Analysis
# Exclude non-numeric columns before calculating mean
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
segment_summary = df.groupby('Segment')[numeric_columns].mean()
print(segment_summary)

# Visualize Segments
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', palette='Set2')
plt.title('Customer Segments')
plt.show()

# Save clustered data
file_path = 'segmented_customers.xlsx'
df.to_excel(file_path, index=False)

# Check if the file was created
if os.path.exists(file_path):
    print(f"File '{file_path}' was created successfully!")
else:
    print(f"Failed to create the file '{file_path}'.")