# Lung Cancer Data Analysis

This project uses various machine learning algorithms for regression and clustering to analyze a dataset on lung cancer. The goal is to predict outcomes using regression techniques and uncover patterns in the data using clustering.

## Dataset

The dataset `lung_cancer.csv` includes clinical features of patients, which are used as input variables for the models.

## Algorithms Used

1. **Linear Regression**: Fits a linear model to predict outcomes based on feature data.
2. **Decision Tree Regression**: Divides the dataset into subsets using the most significant features.
3. **SVM Regression**: Employs Support Vector Machines for regression tasks.
4. **DBSCAN Clustering**: Groups data points based on density, identifying noise and clusters.
5. **K-Means Clustering**: Partitions data into K distinct clusters for analysis.

## Code Structure

1. **Preprocessing**:
   - Removes irrelevant columns and maps diagnosis labels (`M` as 1, `B` as 0).
   - Scales data for clustering algorithms.

2. **Regression**:
   - Compares the performance of Linear Regression, Decision Tree Regression, and SVM Regression.
   - Uses metrics like MAE, MSE, and R² to evaluate models.

3. **Clustering**:
   - DBSCAN identifies density-based clusters and noise points.
   - K-Means evaluates clustering performance across different K values using multiple metrics.

## How to Run

1. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

2. Change file path:

dataframe = pd.read_csv(">> SET YOUR FILE PATH HERE <<", encoding="latin1")

Make sure to replace `>> SET YOUR FILE PATH HERE <<` with the full file path to your dataset. 

Note: If you copy the file path directly from your file explorer, it might use backslashes (`\`) instead of forward slashes (`/`). Replace backslashes with forward slashes to avoid errors.

Example:

```python
dataframe = pd.read_csv("C:/Users/YourUser/Documents/dataset/lung_cancer.csv", encoding="latin1")
```

3. Run the code:
   ```bash
   python data_mining.py
   ```

## Conclusion

This project demonstrates the application of regression and clustering techniques on real-world medical data, showcasing the importance of preprocessing, evaluation, and model selection for effective analysis.