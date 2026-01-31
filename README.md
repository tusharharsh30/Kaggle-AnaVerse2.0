# Kaggle Anaverse 2.0 – Anomaly Detection

## Problem Overview
This project focuses on detecting anomalies from multivariate sensor readings collected at regular time intervals from an energy manufacturing plant.

The data consists of timestamped sensor measurements (`X1`–`X5`) and a binary target variable:
- **0** → Normal operation  
- **1** → Anomalous behavior  

The task is challenging due to extreme class imbalance, temporal dependencies between observations, and subtle anomaly patterns that overlap with normal behavior.

---

## Key Challenges
- **Severe class imbalance** (anomalies < 1% of data)  
- **Time-dependent structure**, requiring causal modeling  
- **Overlapping distributions** between normal and anomalous states  

As a result, traditional accuracy-based approaches are ineffective, and F1-score is used as the primary evaluation metric.

---

## Solution Approach
The solution follows a clean, end-to-end machine learning workflow designed for real-world deployment:

- Temporal feature extraction using lagged values and rolling statistics  
- Identification of operational regimes using clustering  
- Time-based train–validation split to prevent data leakage  
- Gradient boosting model optimized for highly imbalanced data  
- Threshold tuning to directly maximize F1-score  

> Detailed reasoning, visualizations, and feature analysis are documented extensively in the notebook.

---

## Model
A **LightGBM** classifier is used due to its strong performance on large tabular datasets and ability to capture non-linear feature interactions.  
Class imbalance is handled internally, and decision thresholds are tuned on a validation set.

---

## Results
- **Final Score:** 0.723978086  
- **Leaderboard Rank:** 231 / 860 (Top ~27%)

---

## Dataset
The dataset is provided by the Kaggle Anaverse 2.0 competition and is stored in **Parquet format** for efficient processing.

Dataset files are not included in this repository.  
Please download them from Kaggle and place them inside the `data/` directory before running the notebook.

---


## Repository Structure
```plaintext
kaggle-anaverse-2.0/
├── notebook.ipynb
├── README.md
├── requirements.txt
├── data/        # not included (Kaggle dataset files)
└── output/      # not included (generated submission files)
```

---

## Notes
- The notebook was originally developed on Kaggle and adapted for local execution  
- The focus of this project is on **robust, causal anomaly detection** rather than leaderboard-only optimization  

