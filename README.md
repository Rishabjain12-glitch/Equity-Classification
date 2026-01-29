# Equity Classification: Smart Investing with Machine Learning

Welcome to the **Equity Classification** project. This tool is designed to have a rough idea if a company is investable or not.

By using Machine Learning to analyze financial ratios, this project helps identify "Good" vs. "Bad" long-term investments with a level of precision that traditional "human-only" analysis often misses.

---

## Project Goal
The objective is simple: **Predict the long-term investment quality of a company.**

Unlike algorithmic trading that looks for short-term price patterns, this project focuses on **fundamental ratios** (like ROE, Debt-to-Equity, and Profit Margins) to determine if a company is built for sustainable growth.

---

## Key Features

### 1. The "Safety First" Approach
In the world of investing, losing money (a **False Positive**) is far worse than missing a gain. Our models are specifically tuned to minimize the risk of classifying a weak company as "Good."

### 2. Machine Learning vs. Tradition
We don't just use AI; we compare it against a **Conventional Rule-Based Model**.
- **Conventional Logic**: Standard benchmarks (e.g., "Is ROE > 15%?").

### 3. Comprehensive Model Suite
We evaluate 10+ different types of models to find the best performer, including:
- **Random Forests & Gradient Boosting** (Excellent for complex patterns)
- **Neural Networks** (For deep data relationships)
- **Logistic Regression & SVM** (For robust classification)

---

## How It Works

1.  **Data Ingestion**: Processes core financial statements (Income Statement & Balance Sheet).
2.  **Smart Ratios**: Computes 25+ essential ratios covering Profitability, Liquidity, Solvency, and Efficiency.
3.  **Feature Selection**: Automatically removes redundant data to focus on the real "Alpha" drivers.
4.  **Optimization**: Uses Grid Search to fine-tune the best-performing models for maximum accuracy.



---

## Key Insights
Our analysis consistently shows that **Profitability Ratios** (like ROE and Net Margin) combined with **Solvency Ratios** (Interest Coverage) are the strongest predictors of long-term success. Machine Learning allows us to see how these factors interact in ways that simple screening tools cannot.

