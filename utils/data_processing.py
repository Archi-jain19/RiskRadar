"""
DropGuard AI — Data Processing Utilities
==========================================
Shared utilities for data loading, cleaning, sequence preparation, and splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset with basic validation.
    
    Args:
        path: Absolute or relative path to the CSV file.
        
    Returns:
        DataFrame with the loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or not a valid CSV.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset at {path} is empty.")
        print(f"  ✓ Loaded {len(df)} records with {len(df.columns)} columns from {path}")
        return df
    except Exception as e:
        raise type(e)(f"Error loading dataset: {e}")


def clean_data(df: pd.DataFrame, target_col: str = 'Dropout') -> pd.DataFrame:
    """Clean dataset: handle missing values, remove duplicates, clip outliers.
    
    Strategy:
      - Numeric columns: KNN imputation (k=5) for columns with <30% missing,
        median fill for columns with ≥30% missing.
      - Categorical columns: mode fill.
      - Duplicate rows: removed.
      
    Args:
        df: Raw DataFrame.
        target_col: Name of the target column to exclude from imputation.
        
    Returns:
        Cleaned DataFrame.
    """
    cleaned = df.copy()
    
    # Remove exact duplicate rows
    n_dupes = cleaned.duplicated().sum()
    if n_dupes > 0:
        cleaned = cleaned.drop_duplicates()
        print(f"  ✓ Removed {n_dupes} duplicate rows")
    
    # Separate numeric and categorical columns
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Exclude target from imputation
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Numeric imputation
    cols_with_missing = [c for c in numeric_cols if cleaned[c].isnull().any()]
    if cols_with_missing:
        # KNN imputation for columns with reasonable missing rates
        knn_cols = [c for c in cols_with_missing if cleaned[c].isnull().mean() < 0.3]
        median_cols = [c for c in cols_with_missing if cleaned[c].isnull().mean() >= 0.3]
        
        if knn_cols:
            imputer = KNNImputer(n_neighbors=5)
            cleaned[knn_cols] = imputer.fit_transform(cleaned[knn_cols])
            print(f"  ✓ KNN-imputed {len(knn_cols)} numeric columns")
        
        for col in median_cols:
            median_val = cleaned[col].median()
            cleaned[col].fillna(median_val, inplace=True)
            print(f"  ✓ Median-filled {col} ({cleaned[col].isnull().sum()} remaining nulls)")
    
    # Categorical imputation
    for col in categorical_cols:
        if cleaned[col].isnull().any():
            mode_val = cleaned[col].mode()[0]
            cleaned[col].fillna(mode_val, inplace=True)
            print(f"  ✓ Mode-filled {col} with '{mode_val}'")
    
    print(f"  ✓ Cleaning complete: {len(cleaned)} rows, {cleaned.isnull().sum().sum()} nulls remaining")
    return cleaned


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    student_id_col: str = 'Student_ID',
    semester_col: str = 'Semester',
    target_col: str = 'Dropout',
    max_seq_len: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flat DataFrame to 3D sequences for LSTM input.
    
    Groups data by student, sorts by semester, pads/truncates to max_seq_len.
    
    Args:
        df: DataFrame with multi-semester student records.
        feature_cols: List of feature column names to include in sequences.
        student_id_col: Column identifying unique students.
        semester_col: Column for temporal ordering.
        target_col: Target variable column.
        max_seq_len: Maximum sequence length (pad/truncate to this length).
        
    Returns:
        Tuple of (X_sequences, y_labels, student_ids):
          - X_sequences: np.ndarray of shape (n_students, max_seq_len, n_features)
          - y_labels: np.ndarray of shape (n_students,)
          - student_ids: np.ndarray of student IDs
    """
    sequences = []
    labels = []
    student_ids = []
    
    grouped = df.sort_values([student_id_col, semester_col]).groupby(student_id_col)
    
    for sid, group in grouped:
        # Extract feature matrix for this student's semesters
        seq = group[feature_cols].values.astype(np.float32)
        
        # Target: use the last known label for this student
        label = group[target_col].iloc[-1]
        
        # Pad or truncate
        if len(seq) >= max_seq_len:
            seq = seq[-max_seq_len:]  # Take last N semesters
        else:
            # Zero-pad at the beginning
            pad_len = max_seq_len - len(seq)
            padding = np.zeros((pad_len, len(feature_cols)), dtype=np.float32)
            seq = np.vstack([padding, seq])
        
        sequences.append(seq)
        labels.append(label)
        student_ids.append(sid)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    ids = np.array(student_ids)
    
    print(f"  ✓ Prepared {len(X)} sequences: shape {X.shape}")
    return X, y, ids


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> dict:
    """Stratified train/validation/test split.
    
    Args:
        X: Feature matrix (2D or 3D).
        y: Target vector.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        random_state: Random seed.
        
    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=random_state, stratify=y_temp
    )
    
    print(f"  ✓ Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }
