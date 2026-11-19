"""
Real-World Dataset Analysis Project
Dataset: Iris flower measurements (built into scikit-learn)
Run: python real_world_dataset_analysis.py

Install requirements:
pip install pandas matplotlib seaborn scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def main():
    # Load a real-world dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=["target"])

    print("\n--- Dataset Overview ---")
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe())

    # Data cleaning/preprocessing
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Basic exploratory analysis
    print("\nSpecies counts:")
    print(df["species"].value_counts())
    print("\nAverage measurements by species:")
    print(df.groupby("species")[numeric_cols].mean().round(2))

    # Correlation analysis
    correlation = df[numeric_cols].corr()
    print("\nCorrelation matrix:")
    print(correlation.round(2))

    # Standardize numeric features (demonstration of preprocessing)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
                         columns=numeric_cols)
    print("\nStandardized data preview:")
    print(scaled.head().round(2))

    # Visualizations
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="sepal length (cm)", hue="species",
                 kde=True, element="step")
    plt.title("Sepal Length Distribution by Species")
    plt.tight_layout()
    plt.savefig("sepal_length_distribution.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)",
                   hue="species")
    plt.title("Petal Length vs Petal Width")
    plt.tight_layout()
    plt.savefig("petal_relationship.png", dpi=150)
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()

    # Export cleaned dataset
    df.to_csv("iris_cleaned_dataset.csv", index=False)
    print("\nSaved cleaned dataset as iris_cleaned_dataset.csv")
    print("Saved plots: sepal_length_distribution.png, petal_relationship.png, correlation_heatmap.png")

if __name__ == "__main__":
    main()

