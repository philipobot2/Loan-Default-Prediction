# Loan Default Prediction Project

## Overview
This project aims to build a predictive model to identify loan applicants who are likely to default on their loans. By accurately predicting loan defaults, financial institutions can mitigate risks, optimize lending decisions, and reduce potential financial losses. This is a binary classification problem where the target variable 'Default' indicates whether a loan was defaulted (1) or not (0).

## Data Analysis Key Findings

*   **Problem Definition**: The primary goal was to predict loan defaults, focusing on high accuracy, precision, and recall for the 'Default' class to effectively manage risk.
*   **Data Quality**: No duplicate rows were found in the dataset, indicating a clean initial dataset structure.
*   **Data Preprocessing**: Outliers in numerical features were handled using the Interquartile Range (IQR) method with capping to ensure robust model training. The 'LoanID' column was removed as it served merely as an identifier.
*   **Categorical Feature Handling**: All categorical columns were successfully transformed using one-hot encoding, making them suitable for machine learning models.
*   **Dataset Imbalance**: Exploratory Data Analysis (EDA) revealed that the dataset is imbalanced, with a significantly smaller proportion of defaulted loans compared to non-defaulted ones. This was a critical factor influencing model selection and evaluation strategy.
*   **Feature Correlations**: Numerical features generally showed weak correlations with the 'Default' target variable. Notably, 'InterestRate' displayed a slight positive correlation (+0.131), while 'Age' (-0.168), 'Income' (-0.099), and 'MonthsEmployed' (-0.097) exhibited slight negative correlations.
*   **Feature Engineering**: Three new features were engineered to capture more complex relationships and potentially enhance model performance:
    *   `DTIRatio_LoanAmount`: Interaction between Debt-to-Income Ratio and Loan Amount.
    *   `CreditScore_InterestRate_Interaction`: Interaction between Credit Score and Interest Rate.
    *   `LoanAmount_Income_Ratio`: Ratio of Loan Amount to Income.

## Model Selection and Training

Based on the binary classification nature of the problem, the large dataset size, and the mixed types of features, the following models were selected and trained:

*   **Logistic Regression**: Chosen as a strong baseline due to its interpretability, computational efficiency, and effectiveness with numerical and one-hot encoded features.
*   **Random Forest**: An ensemble method capable of capturing complex non-linear relationships and interactions, robust to various data types, and less prone to overfitting than single decision trees.
*   **LightGBM**: A highly efficient and accurate gradient boosting framework, particularly well-suited for large datasets and known for its speed and ability to handle mixed feature types.

All models were trained on an 80% training set, with 20% reserved for testing.

## Model Performance Summary

The models were evaluated on the unseen test set using several key metrics, recognizing the importance of precision, recall, and F1-score for imbalanced datasets.

| Model                  | Accuracy | Precision | Recall  | F1-Score | ROC AUC |
|:-----------------------|:---------|:----------|:--------|:---------|:--------|
| Logistic Regression    | 0.8837   | 0.6974    | 0.1691  | 0.2729   | 0.7303  |
| Random Forest          | 0.8872   | 0.7410    | 0.1904  | 0.3031   | 0.7719  |
| LightGBM               | 0.8885   | 0.7483    | 0.2001  | 0.3151   | 0.7816  |

**Comparison Summary:**

*   **Accuracy** provides a general measure of correctness, but for imbalanced datasets like this, it can be misleading.
*   **Precision** indicates the proportion of positive identifications that were actually correct. High precision is vital to avoid incorrectly classifying non-defaulters as defaulters, which could lead to unnecessary loan rejections.
*   **Recall** measures the proportion of actual positives that were correctly identified. High recall is critical for identifying as many actual defaulters as possible, thereby minimizing missed risks.
*   **F1-Score** is the harmonic mean of precision and recall, offering a balanced measure, especially important when both false positives and false negatives have significant costs.
*   **ROC AUC** measures the model's ability to distinguish between the two classes across various thresholds.

Based on the evaluation metrics, **LightGBM emerged as the best-performing model**, demonstrating the highest F1-Score and ROC AUC, indicating a superior ability to handle the imbalanced nature of the dataset and effectively distinguish between defaulting and non-defaulting loans.

## Actionable Insights

*   **Interest Rate Impact**: The slight positive correlation of `InterestRate` with default suggests that higher interest rates might correspond to a higher likelihood of default. This could be due to borrowers with higher risk profiles being offered higher rates, or higher rates increasing the burden on borrowers.
*   **Demographic and Employment Factors**: Slight negative correlations with `Age`, `Income`, and `MonthsEmployed` indicate that younger, lower-income, and less-employed individuals might have a marginally higher propensity to default. This aligns with general financial risk principles.
*   **Feature Engineering Effectiveness**: The engineered features, particularly the interaction terms, likely contributed to LightGBM's superior performance by providing the model with more nuanced information about borrower characteristics.
*   **Imbalanced Data**: The dataset's imbalance highlights the importance of using appropriate evaluation metrics (like Precision, Recall, F1-Score, and ROC AUC) beyond simple accuracy, and considering techniques to address class imbalance if further performance gains are needed (e.g., SMOTE, undersampling).

## Recommendations

1.  **Deployment of LightGBM**: The trained LightGBM model is recommended for deployment. It provides the best balance of predictive power for identifying loan defaults and can be integrated into existing loan application processing systems.
2.  **Threshold Optimization**: Financial institutions should carefully determine the optimal probability threshold for classifying a loan as 'default'. This threshold should be set based on their specific risk appetite and the cost associated with false positives (rejecting a creditworthy applicant) versus false negatives (approving a defaulting loan).
3.  **Continuous Monitoring**: Implement a robust monitoring system for the deployed model. This system should track key performance metrics on new, incoming data to detect 'model drift' caused by changing economic conditions, new lending policies, or evolving borrower behaviors.
4.  **Regular Retraining**: To maintain the model's effectiveness and adapt to dynamic market conditions, periodic retraining with fresh, updated data is crucial. This will ensure the model remains accurate and relevant over time.
5.  **Further Feature Engineering**: Explore additional feature engineering opportunities, possibly incorporating external economic indicators or more complex interactions, to further enhance predictive performance.
6.  **Explainable AI (XAI)**: Implement Explainable AI techniques to understand why the LightGBM model makes certain predictions. This can provide valuable insights into the primary drivers of loan default and help build trust with stakeholders and regulators.
