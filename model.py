"""
Fraud Transaction Detection with PyTorch
Complete pipeline: Feature Engineering + Neural Network Classifier
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (classification_report, precision_recall_curve, 
                            roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import pickle
import zipfile
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================

class FraudDataProcessor:
    """Handles loading, cleaning, and feature engineering"""
    
    def __init__(self, zip_path='dataset.zip'):
        self.zip_path = zip_path
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all daily pickle files from zip"""
        print("Loading data from zip file...")
        all_dfs = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            pickle_files = [f for f in zip_ref.namelist() if f.endswith('.pkl')]
            
            for i, pkl_file in enumerate(pickle_files):
                with zip_ref.open(pkl_file) as f:
                    df_day = pickle.load(f)
                    all_dfs.append(df_day)
                    
                if (i + 1) % 20 == 0:
                    print(f"Loaded {i + 1}/{len(pickle_files)} files")
        
        self.df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total transactions loaded: {len(self.df):,}")
        print(f"Fraud rate: {self.df['TX_FRAUD'].mean()*100:.4f}%")
        return self
    
    def engineer_features(self):
        """Create time-based and aggregation features"""
        print("\nEngineering features...")
        df = self.df.copy()
        
        # Convert datetime
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        
        # Basic datetime features
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['weekday'] = df['TX_DATETIME'].dt.weekday
        df['day'] = df['TX_DATETIME'].dt.day
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Amount features
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        df['amount_gt_220'] = (df['TX_AMOUNT'] > 220).astype(int)
        
        # Set datetime as index for rolling operations
        df = df.set_index('TX_DATETIME')
        
        # Customer rolling features
        print("Computing customer rolling features...")
        for window in ['1D', '7D', '14D', '28D']:
            window_name = window.replace('D', 'd')
            
            # Transaction counts
            df[f'cust_tx_count_{window_name}'] = (
                df.groupby('CUSTOMER_ID')['TX_AMOUNT']
                .rolling(window, min_periods=1).count()
                .reset_index(level=0, drop=True)
            )
            
            # Amount statistics
            df[f'cust_tx_amt_mean_{window_name}'] = (
                df.groupby('CUSTOMER_ID')['TX_AMOUNT']
                .rolling(window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            
            df[f'cust_tx_amt_std_{window_name}'] = (
                df.groupby('CUSTOMER_ID')['TX_AMOUNT']
                .rolling(window, min_periods=1).std()
                .reset_index(level=0, drop=True)
            ).fillna(0)
        
        # Terminal rolling features
        print("Computing terminal rolling features...")
        for window in ['7D', '28D']:
            window_name = window.replace('D', 'd')
            
            df[f'terminal_tx_count_{window_name}'] = (
                df.groupby('TERMINAL_ID')['TX_AMOUNT']
                .rolling(window, min_periods=1).count()
                .reset_index(level=0, drop=True)
            )
        
        # Reset index
        df = df.reset_index()
        
        # Derived features
        df['amount_to_cust_mean_ratio_14d'] = (
            df['TX_AMOUNT'] / (df['cust_tx_amt_mean_14d'] + 1e-5)
        )
        
        df['amount_to_cust_std_ratio_14d'] = (
            df['TX_AMOUNT'] / (df['cust_tx_amt_std_14d'] + 1e-5)
        )
        
        # Fill any remaining NaNs
        df = df.fillna(0)
        
        self.df = df
        print("Feature engineering complete!")
        return self
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        """Time-based split (critical for fraud detection)"""
        df = self.df.sort_values('TX_DATETIME')
        
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        print(f"\nData split:")
        print(f"Train: {len(train_df):,} ({train_df['TX_FRAUD'].sum():,} frauds)")
        print(f"Val:   {len(val_df):,} ({val_df['TX_FRAUD'].sum():,} frauds)")
        print(f"Test:  {len(test_df):,} ({test_df['TX_FRAUD'].sum():,} frauds)")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df, fit_scaler=False):
        """Extract and normalize features"""
        feature_cols = [
            'TX_AMOUNT', 'amount_log', 'amount_gt_220',
            'hour', 'weekday', 'day', 'is_weekend', 'is_night',
            'cust_tx_count_1d', 'cust_tx_count_7d', 'cust_tx_count_14d', 'cust_tx_count_28d',
            'cust_tx_amt_mean_1d', 'cust_tx_amt_mean_7d', 'cust_tx_amt_mean_14d', 'cust_tx_amt_mean_28d',
            'cust_tx_amt_std_1d', 'cust_tx_amt_std_7d', 'cust_tx_amt_std_14d', 'cust_tx_amt_std_28d',
            'terminal_tx_count_7d', 'terminal_tx_count_28d',
            'amount_to_cust_mean_ratio_14d', 'amount_to_cust_std_ratio_14d'
        ]
        
        X = df[feature_cols].values
        y = df['TX_FRAUD'].values
        
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y, feature_cols


# ============================================================================
# 2. PYTORCH DATASET AND MODEL
# ============================================================================

class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FraudDetectionNet(nn.Module):
    """Neural Network for fraud detection with residual connections"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(FraudDetectionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# ============================================================================
# 3. TRAINING UTILITIES
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    all_preds = (all_probs >= best_threshold).astype(int)
    
    return all_preds, all_probs, all_labels, best_threshold


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process data
    processor = FraudDataProcessor('dataset.zip')
    processor.load_data().engineer_features()
    
    train_df, val_df, test_df = processor.split_data()
    
    # Prepare features
    X_train, y_train, feature_names = processor.prepare_features(train_df, fit_scaler=True)
    X_val, y_val, _ = processor.prepare_features(val_df, fit_scaler=False)
    X_test, y_test, _ = processor.prepare_features(test_df, fit_scaler=False)
    
    print(f"\nFeature dimension: {X_train.shape[1]}")
    
    # Create datasets
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)
    
    # Handle class imbalance with weighted sampling
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Initialize model
    model = FraudDetectionNet(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_preds, val_probs, val_labels, threshold = evaluate(model, val_loader, device)
        
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auprc = average_precision_score(val_labels, val_probs)
        
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | "
              f"F1: {val_f1:.4f} | AUPRC: {val_auprc:.4f} | "
              f"Threshold: {threshold:.4f}")
        
        scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_fraud_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered")
                break
    
    # Load best model and evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    model.load_state_dict(torch.load('best_fraud_model.pt'))
    test_preds, test_probs, test_labels, test_threshold = evaluate(model, test_loader, device)
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Legitimate', 'Fraud']))
    
    print(f"\nAdditional Metrics:")
    print(f"ROC-AUC: {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"AUPRC: {average_precision_score(test_labels, test_probs):.4f}")
    print(f"Optimal Threshold: {test_threshold:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    print(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    return model, processor, feature_names


if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, f1_score
    model, processor, feature_names = main()