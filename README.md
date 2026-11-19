# Real-World-Dataset-Analysis-Project
This project analyzes a real-world dataset using Python, covering data cleaning, preprocessing, exploratory data analysis (EDA), statistical insights, visualizations, and interpretation of findings.
Real-World Dataset Analysis Project
Overview
This project performs exploratory data analysis (EDA), cleaning, preprocessing, summary statistics, correlation analysis, and visualization using the Iris flower dataset.
Dataset
The Iris dataset is included with scikit-learn. It contains measurements of 150 flowers from three species:
- Setosa
- Versicolor
- Virginica
Features include sepal length, sepal width, petal length, and petal width.
Requirements
Install Python 3.8+ and run:
pip install pandas matplotlib seaborn scikit-learn
How to Run
1. Save real_world_dataset_analysis.py to a folder.
2. Open Command Prompt/Terminal in that folder.
3. Run:
python real_world_dataset_analysis.py
If python is not recognized, try py real_world_dataset_analysis.py.
What the script does
- Loads and inspects the dataset
- Checks data types and missing values
- Removes duplicate rows and fills missing numeric values with medians
- Displays descriptive statistics and species counts
- Compares average measurements by species
- Calculates correlations
- Standardizes numeric features
- Creates and saves three charts
- Exports the cleaned data to CSV
Output Files
After running, it creates:
- iris_cleaned_dataset.csv
- sepal_length_distribution.png
- petal_relationship.png
- correlation_heatmap.png
Key questions
- How do measurements vary among flower species?
- Which flower features are strongly correlated?
- Can petal measurements help distinguish species?
