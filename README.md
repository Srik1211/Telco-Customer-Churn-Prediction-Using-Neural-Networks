# Telco Customer Churn Prediction Using Neural Networks

A machine learning project that uses a **Multi-Layer Perceptron (MLP) neural network** to predict customer churn in a telecommunications dataset.

The project covers the complete workflow from **data cleaning and feature engineering to exploratory data analysis, neural-network modelling, performance evaluation, and business-oriented retention recommendations**.

---

## Project Overview

Customer churn prediction helps telecommunications companies identify customers who are at risk of leaving and enables targeted retention strategies.

This project uses the **Telco Customer Churn dataset**, containing **7,043 customer records**, to build a binary classification model that predicts whether a customer will churn.

The workflow includes:

```text
Customer Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Encoding & Scaling
      ↓
Exploratory Data Analysis
      ↓
Neural Network
      ↓
Model Evaluation
      ↓
Churn Risk Identification
      ↓
Business Retention Decisions
```

---

## Project Objectives

The main objectives are to:

- Understand the characteristics of telecom customers.
- Identify factors associated with customer churn.
- Prepare categorical and numerical variables for neural-network modelling.
- Address class imbalance during model training.
- Build a feedforward neural network using an MLP architecture.
- Evaluate the model using appropriate classification metrics.
- Translate model and EDA findings into practical customer-retention strategies.

---

## Dataset

The project uses the **Telco Customer Churn dataset** containing:

| Property | Details |
|---|---|
| Total Customers | 7,043 |
| Target Variable | `Churn` |
| Prediction Type | Binary Classification |
| Churn Classes | Yes / No |
| Features | Customer demographics, services, contract and billing information |

The dataset contains information about:

- Customer demographics
- Partner and dependent status
- Phone services
- Internet services
- Streaming services
- Security and support services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer tenure

---

## Data Preparation

### Data Cleaning

Two important data-quality issues were addressed:

### `customerID`

The customer ID is a unique identifier and does not provide predictive information, so it was removed from the modelling dataset.

### `TotalCharges`

`TotalCharges` was initially stored as an object because of whitespace values associated with customers who had zero tenure.

The column was converted to numeric and the resulting missing values were replaced with `0`, representing customers who had not yet accumulated charges.

---

## Feature Engineering

The dataset contains both binary and categorical variables.

### Binary Encoding

Yes/No variables were converted to numerical values:

```text
Yes → 1
No  → 0
```

The same approach was applied to features such as:

- Partner
- Dependents
- PhoneService
- PaperlessBilling
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

Gender was also encoded numerically.

### One-Hot Encoding

Multi-class categorical variables were one-hot encoded:

- InternetService
- Contract
- PaymentMethod
- MultipleLines

`drop_first=True` was used to avoid redundant dummy variables.

---

## Train/Test Split & Scaling

The dataset is divided using an **80/20 stratified train/test split**.

Stratification ensures that the proportion of churned and non-churned customers remains similar in both datasets.

### Feature Scaling

`StandardScaler` is applied before training the neural network.

Scaling transforms the features to approximately:

```text
Mean = 0
Standard Deviation = 1
```

This is particularly important for neural networks because features with larger numerical ranges could otherwise dominate the optimization process.

---

# Exploratory Data Analysis

The project performs EDA to identify important patterns associated with customer churn.

---

## Insight 1 — Contract Type and Churn

Contract type is one of the strongest churn-related variables in the dataset.

The analysis shows approximately:

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | ~42% |
| One year | ~11% |
| Two year | ~3% |

Month-to-month customers therefore represent a significantly higher-risk segment compared with customers on longer contracts.

### Business Implication

Contract-based retention strategies can be explored for high-risk month-to-month customers, particularly where customers show additional signs of churn risk.

---

## Insight 2 — Customer Tenure

The analysis compares tenure distributions between churned and retained customers.

Key observation:

- Churned customers have an average tenure of approximately **18 months**.
- Retained customers have an average tenure of approximately **37 months**.
- Churn is particularly concentrated among customers during the earlier stages of their relationship.

### Business Implication

The first several months of the customer lifecycle represent an important opportunity for:

- Customer onboarding
- Engagement
- Customer support
- Product education
- Early retention initiatives

---

## Insight 3 — Fiber Optic Customers

The analysis identifies a relatively high churn rate among Fiber Optic customers.

The churn rate is approximately:

- Fiber Optic: **~41%**
- DSL: **~19%**

Fiber Optic customers also tend to have higher monthly charges.

### Business Implication

The combination of higher pricing and higher churn suggests that the Fiber Optic customer experience warrants further investigation.

Potential areas include:

- Service reliability
- Network quality
- Technical support
- Customer expectations
- Pricing/value perception

The analysis alone identifies an association rather than establishing a causal relationship.

---

## Correlation Analysis

The project also examines relationships between numerical variables:

- Tenure
- MonthlyCharges
- TotalCharges
- Churn

One notable observation is the negative relationship between tenure and churn, indicating that longer-tenured customers tend to have lower churn rates.

`TotalCharges` is also strongly related to tenure and monthly charges because total charges accumulate over the customer relationship.

---

# Neural Network Model

The project uses a **Multi-Layer Perceptron (MLP)** implemented using Scikit-learn's `MLPClassifier`.

### Architecture

```text
Input Layer
26 Features
     ↓
Hidden Layer 1
64 Neurons
ReLU
     ↓
Hidden Layer 2
32 Neurons
ReLU
     ↓
Output Layer
1 Neuron
Binary Classification
```

### Model Configuration

| Parameter | Value |
|---|---|
| Model | MLPClassifier |
| Hidden Layers | 64 → 32 |
| Activation | ReLU |
| Optimizer | Adam |
| Maximum Iterations | 300 |
| Early Stopping | Enabled |
| Validation Fraction | 10% |
| No Improvement | 15 iterations |
| Random State | 42 |

---

## Handling Class Imbalance

The dataset contains approximately:

```text
73.5% → No Churn
26.5% → Churn
```

A model that predicts every customer as "No Churn" could achieve approximately 73.5% accuracy while failing to identify actual churners.

Therefore, accuracy alone is not sufficient.

The project uses **sample weighting** to give greater importance to the minority churn class during model training.

---

# Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- AUC-ROC
- ROC Curve

### Why Recall Matters

For churn prediction, identifying customers who are actually going to leave is particularly important.

A **False Negative** represents a customer who was likely to churn but was not identified by the model.

Therefore, churn-class recall is an important metric for evaluating the usefulness of the model in a retention setting.

---

## Training Loss

The project visualizes the MLP training loss over iterations.

The loss curve helps examine:

- Training convergence
- Optimization behaviour
- Whether training is continuing to improve
- The effect of early stopping

---

## ROC-AUC

The project calculates the **ROC-AUC score** to evaluate the model's ability to distinguish between churned and retained customers across different classification thresholds.

The notebook reports an AUC-ROC above **0.80**, indicating strong discriminatory performance on the evaluated test data.

---

# Business Decisions

The project translates the analysis and model results into two potential retention strategies.

---

## Decision 1 — Proactive Retention for High-Risk Month-to-Month Customers

The proposed approach is to periodically score active customers using their predicted churn probability.

Customers meeting a specified risk threshold can be considered for targeted retention outreach.

The notebook uses:

```text
Predicted Churn Probability > 0.6
```

as an initial example threshold.

Potential retention actions include:

- Personalized outreach
- Contract upgrade offers
- Service bundle offers
- Targeted discounts

The threshold should ultimately be calibrated using the actual business cost of false positives versus missed churners.

---

## Decision 2 — Fiber Optic Service Quality Investigation

The analysis identifies unusually high churn among Fiber Optic customers.

A proposed business response is to investigate:

- Network/service quality
- Technical support
- Customer complaints
- Reliability issues
- Pricing/value perception

Potential interventions can include targeted support and relevant service bundles.

Importantly, the analysis identifies correlations and patterns; further operational data would be required to establish the underlying causes of churn.

---

# Model Limitations

The project also identifies several limitations.

### Static Dataset

The dataset represents a historical snapshot. Customer behaviour and market conditions can change over time.

### Missing Causal Variables

The dataset does not include variables such as:

- Complaint history
- Network quality
- Service outage data
- Competitor pricing
- Customer satisfaction scores

Therefore, the model identifies predictive patterns rather than causal explanations.

### Precision-Recall Trade-off

Increasing churn recall can increase false positives.

A business must determine the acceptable trade-off between:

```text
Cost of unnecessary retention offers
                vs.
Cost of losing a customer
```

### Threshold Selection

The default classification threshold should not automatically be considered optimal for business deployment.

Threshold tuning should be based on actual retention costs and customer lifetime value.

---

# Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- MLPClassifier
- StandardScaler
- Train/Test Split
- Sample Weighting

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook

---

# Project Structure

```text
telco-churn-analysis/
│
├── telco_churn_analysis.ipynb
├── telco_churn.csv
│
├── fig_churn_distribution.png
├── fig_contract_churn.png
├── fig_tenure_churn.png
├── fig_internet_churn.png
├── fig_correlation.png
├── fig_loss_curve.png
├── fig_evaluation.png
│
└── README.md
```
# Key Learning Outcomes

This project demonstrates practical experience with:

- End-to-end classification workflows
- Customer churn analysis
- Data cleaning
- Feature engineering
- Categorical encoding
- Feature scaling
- Exploratory data analysis
- Neural networks
- Multi-Layer Perceptrons
- ReLU activation
- Adam optimization
- Early stopping
- Class imbalance handling
- Confusion matrices
- ROC curves
- AUC-ROC
- Business-oriented model interpretation
- Translating ML results into retention strategies

---

# End-to-End Workflow

```text
Raw Telco Dataset
       ↓
Data Cleaning
       ↓
Feature Encoding
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Exploratory Data Analysis
       ↓
MLP Neural Network
       ↓
Sample Weighting
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Churn Risk Analysis
       ↓
Business Retention Strategy
```

---

#  Conclusion

This project demonstrates how a neural-network-based churn prediction system can combine **machine learning, exploratory analysis, and business reasoning**.

The analysis identifies contract type, customer tenure, and internet service type as important churn-related patterns and uses these findings alongside model predictions to formulate targeted retention strategies.

The project also highlights an important principle in applied machine learning: **a good predictive model is only part of the solution**. The model's outputs need to be interpreted in the context of business costs, customer behaviour, operational constraints, and potential limitations in the available data.



