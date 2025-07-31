#Step 1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print(iris_df.head())

# Check the data types and missing values
print(iris_df.info())
print(iris_df.isnull().sum())

# Clean the dataset (though Iris dataset typically has no missing values)
iris_df.dropna(inplace=True)





#Step 2: Basic Data Analysis
# Compute basic statistics
print(iris_df.describe())

# Group by species and compute the mean of each numerical column
grouped = iris_df.groupby('species').mean()
print(grouped)






#Step 3: Data Visualization
# Line chart (though not typical for Iris dataset, we can simulate a trend)
plt.figure(figsize=(10, 6))
for species in iris_df['species'].unique():
    subset = iris_df[iris_df['species'] == species]
    plt.plot(subset['sepal length (cm)'], label=species)
plt.title('Trend of Sepal Length by Species')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# Bar chart showing average petal length per species
plt.figure(figsize=(10, 6))
plt.bar(grouped.index, grouped['petal length (cm)'], color=['blue', 'orange', 'green'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram of a numerical column
plt.figure(figsize=(10, 6))
plt.hist(iris_df['sepal width (cm)'], bins=20, color='purple')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of sepal length vs. petal length
plt.figure(figsize=(10, 6))
for species in iris_df['species'].unique():
    subset = iris_df[iris_df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], label=species)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()



