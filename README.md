Fraud Transaction Detection

A machine learning pipeline for detecting fraudulent transactions from a simulated / real-world transaction dataset.

📂 Repository Structure
File / Folder	Description
dataset.zip	Raw transaction dataset split by day (pickle / CSV files).
model.py	Script defining the model training, evaluation, and inference logic.
best_fraud_model.pt	Serialized / saved best-performing model.
confusion_matrix.png	Visualization of confusion matrix on test dataset.
.gitignore	Standard ignore file to skip outputs, virtualenv, etc.
✅ Features & Approach

Reads in the time-series transaction data.

Performs exploratory data analysis (EDA), data cleaning, and feature engineering (e.g. log-transform on amount, time-based aggregations, customer / terminal rolling stats).

Builds a classification model (LightGBM / similar) to detect fraud (TX_FRAUD label).

Uses time-based train-validation-test splitting to respect temporal order.

Optimizes threshold based on precision / recall trade-offs.

Evaluates model via precision, recall, F1-score, ROC-AUC, AUPRC, confusion matrix.

📈 Evaluation Metrics

When evaluated on the test split, the model reports:

High overall accuracy (≈ 99 %) — but strong class imbalance means accuracy is not enough.

Precision / Recall metrics for fraud class indicate trade-off: high precision but relatively low recall.

AUPRC (Area Under Precision-Recall Curve) used as key metric given the imbalanced fraud detection scenario.

The model optimizes a decision threshold to maximize F1 or other fraud-detection-oriented metric.

🚀 How to Run

Prepare environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Load / preprocess data
Unzip dataset.zip if not already. Ensure model-script paths match your local structure.

Train model

python model.py --train


Evaluate on test set / infer

python model.py --evaluate


Adjust / inspect decision threshold
You can modify threshold values in model.py to trade off precision / recall, or plot a precision-recall curve.

🔧 Customization & Tuning

Modify feature engineering logic (observations / rolling-window lengths).

Tune model hyperparameters (e.g. learning rate, max depth, class weight / scale_pos_weight).

Change CV strategy (e.g. rolling window cross-validation instead of simple hold-out split).

Persist alternative models; compare performance via metrics such as AUPRC, F1 at desired recall levels.

⚠️ Important Notes

Because frauds are very rare in the dataset, accuracy alone is misleading. Focus on precision / recall / F1 and precision-recall curve.

Decision threshold matters: default (0.5) may not be optimal; adjust to business-oriented constraints (e.g. false positive budget).

Avoid data leakage across time windows; always use past-only features for real-time readiness.

🌱 Roadmap & Potential Improvements

Add more sophisticated features:

Customer and terminal historical behavior features.

Time-decay features (recentness weighting).

Graph-based features (customer-terminal network).

Anomaly detection pre-filtering (IsolationForest / Autoencoder) before classifier.

Model ensemble or stacking (e.g. LightGBM + neural net).

Real-time inference / serving pipeline (feature store).

Monitoring / drift detection when deployed in production-like environment.
