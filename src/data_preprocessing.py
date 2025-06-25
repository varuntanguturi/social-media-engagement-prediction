import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def handle_missing_values(df, strategy='mean', columns=None, knn_neighbors=5):
    """
    Handle missing values in the dataframe using different strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing missing values
    strategy : str
        Strategy for imputation: 'mean', 'median', 'mode', 'constant', 'knn'
    columns : list or None
        List of columns to apply imputation to. If None, applies to all columns with missing values.
    knn_neighbors : int
        Number of neighbors to use for KNN imputation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_imputed = df.copy()
    
    # If columns not specified, apply to all columns with missing values
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    # Filter only numeric columns for imputation
    numeric_columns = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    
    # KNN imputation is handled separately
    if strategy == 'knn':
        if numeric_columns:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df_imputed[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        print(f"Applied KNN imputation to columns: {numeric_columns}")
        return df_imputed
    
    # For other strategies, use SimpleImputer
    if numeric_columns:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        print(f"Applied {strategy} imputation to columns: {numeric_columns}")
    
    # Handle categorical columns if any (using most_frequent)
    categorical_columns = [col for col in columns if col not in numeric_columns]
    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
        print(f"Applied most_frequent imputation to categorical columns: {categorical_columns}")
    
    return df_imputed

def scale_features(df, method='standard', columns=None):
    """
    Scale numerical features using different scaling methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing features to scale
    method : str
        Scaling method: 'standard', 'minmax', 'robust'
    columns : list or None
        List of columns to scale. If None, scales all numerical columns.
        
    Returns:
    --------
    pandas.DataFrame, object
        Scaled DataFrame and the scaler object
    """
    # Create a copy of the dataframe
    df_scaled = df.copy()
    
    # If columns not specified, apply to all numerical columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter only numerical columns from the provided list
        columns = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
    
    # Select the appropriate scaler
    if method == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler (mean=0, std=1)")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        print("Using MinMaxScaler (min=0, max=1)")
    elif method == 'robust':
        scaler = RobustScaler()
        print("Using RobustScaler (median=0, IQR=1)")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Apply scaling if there are numerical columns
    if columns:
        df_scaled[columns] = scaler.fit_transform(df[columns])
        print(f"Applied {method} scaling to columns: {columns}")
    
    return df_scaled, scaler

def engineer_time_features(df, hour_column='post_hour', day_column='post_day'):
    """
    Create additional features from time-related columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing time features
    hour_column : str
        Column name containing hour information (0-24)
    day_column : str
        Column name containing day of week information (0-6)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional time-based features
    """
    df_time = df.copy()
    
    # Check if the required columns exist in the dataframe
    time_features_added = False
    
    # Create hour-based features if the column exists
    if hour_column in df.columns:
        # Convert hour to periodic features using sine and cosine transformations
        # This preserves the cyclical nature of time (hour 23 is close to hour 0)
        df_time['hour_sin'] = np.sin(2 * np.pi * df[hour_column] / 24.0)
        df_time['hour_cos'] = np.cos(2 * np.pi * df[hour_column] / 24.0)
        
        # Create time of day categorical feature
        conditions = [
            (df[hour_column] >= 5) & (df[hour_column] < 12),
            (df[hour_column] >= 12) & (df[hour_column] < 17),
            (df[hour_column] >= 17) & (df[hour_column] < 21),
            (df[hour_column] >= 21) | (df[hour_column] < 5)
        ]
        time_categories = ['morning', 'afternoon', 'evening', 'night']
        df_time['time_of_day'] = np.select(conditions, time_categories, default='unknown')
        
        # Create peak hours flag (typically 7-9 AM and 5-7 PM)
        peak_conditions = [
            (df[hour_column] >= 7) & (df[hour_column] <= 9),
            (df[hour_column] >= 17) & (df[hour_column] <= 19)
        ]
        df_time['is_peak_hour'] = np.any(peak_conditions, axis=0).astype(int)
        
        time_features_added = True
    
    # Create day-based features if the column exists
    if day_column in df.columns:
        # Convert day to periodic features using sine and cosine
        df_time['day_sin'] = np.sin(2 * np.pi * df[day_column] / 7.0)
        df_time['day_cos'] = np.cos(2 * np.pi * df[day_column] / 7.0)
        
        # Create weekend flag
        df_time['is_weekend'] = np.where(df[day_column] >= 5, 1, 0)
        
        time_features_added = True
    
    if time_features_added:
        print("Successfully added time-based features")
    else:
        print("No time features were added - required columns not found")
    
    return df_time

def detect_and_handle_outliers(df, columns=None, method='iqr', contamination=0.05, treatment='clip'):
    """
    Detect and handle outliers in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check for outliers
    columns : list or None
        List of columns to check for outliers. If None, uses all numerical columns.
    method : str
        Method to detect outliers: 'iqr' (Interquartile Range) or 'isolation_forest'
    contamination : float
        Expected proportion of outliers in the dataset (for isolation_forest)
    treatment : str
        How to handle outliers: 'clip', 'remove', or 'flag'
        
    Returns:
    --------
    pandas.DataFrame, pandas.DataFrame
        Processed dataframe and dataframe containing information about outliers
    """
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # If columns not specified, check all numerical columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Dictionary to store outlier information
    outlier_info = {col: [] for col in columns}
    outlier_indices = set()
    
    # Detect outliers based on the specified method
    if method == 'iqr':
        for col in columns:
            # Calculate Q1, Q3 and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.update(outliers)
            
            # Store outlier information
            outlier_info[col] = {
                'count': len(outliers),
                'percent': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': outliers.tolist()
            }
            
            # Handle outliers based on treatment method
            if treatment == 'clip' and not outliers.empty:
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
    elif method == 'isolation_forest':
        # Apply Isolation Forest to the selected columns
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(df[columns])
        
        # Identify outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outlier_indices.update(df.index[outlier_mask])
        
        # Store outlier information
        for col in columns:
            col_outliers = df.index[outlier_mask].tolist()
            outlier_info[col] = {
                'count': len(col_outliers),
                'percent': (len(col_outliers) / len(df)) * 100,
                'indices': col_outliers
            }
    
    # Handle outliers based on treatment method
    if treatment == 'remove':
        df_processed = df_processed.drop(list(outlier_indices))
        print(f"Removed {len(outlier_indices)} outlier rows")
    elif treatment == 'flag':
        df_processed['is_outlier'] = df_processed.index.isin(outlier_indices).astype(int)
        print(f"Flagged {len(outlier_indices)} outlier rows")
    
    # Create a summary dataframe of outlier information
    summary = []
    for col in columns:
        if isinstance(outlier_info[col], dict):
            summary.append({
                'column': col,
                'outlier_count': outlier_info[col]['count'],
                'outlier_percent': outlier_info[col]['percent'],
                'treatment': treatment
            })
    
    outlier_summary = pd.DataFrame(summary)
    
    return df_processed, outlier_summary

def encode_categorical_variables(df, columns=None, method='one_hot'):
    """
    Encode categorical variables using specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with categorical variables
    columns : list or None
        List of categorical columns to encode. If None, encodes all object/category columns.
    method : str
        Encoding method: 'one_hot', 'label', or 'ordinal'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    
    # If columns not specified, identify categorical columns
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columns:
        print("No categorical columns found for encoding")
        return df_encoded
    
    # Apply encoding based on specified method
    if method == 'one_hot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        print(f"Applied one-hot encoding to columns: {columns}")
    
    elif method == 'label':
        for col in columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
        print(f"Applied label encoding to columns: {columns}")
    
    elif method == 'ordinal':
        # For ordinal encoding, we would need to define the order of categories
        # This is just a placeholder - in a real scenario, you would define the order
        for col in columns:
            categories = sorted(df_encoded[col].unique())
            cat_map = {cat: i for i, cat in enumerate(categories)}
            df_encoded[col] = df_encoded[col].map(cat_map)
        print(f"Applied ordinal encoding to columns: {columns}")
    
    return df_encoded

def split_dataset(df, target_column, test_size=0.2, val_size=0.0, random_state=None, stratify=None):
    """
    Split the dataset into training, validation, and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to split
    target_column : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    val_size : float
        Proportion of data to use for validation (0.0 to 1.0)
    random_state : int or None
        Seed for random number generation to ensure reproducibility
    stratify : pandas.Series or None
        Data to use for stratified splitting (typically the target for classification)
        
    Returns:
    --------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
        (X_val and y_val will be None if val_size=0.0)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Ensure the specified columns exist
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Set up stratification
    stratify_data = None
    if stratify is not None:
        if isinstance(stratify, str) and stratify in df.columns:
            stratify_data = df[stratify]
        else:
            stratify_data = stratify
    
    # If validation set is requested
    if val_size > 0:
        # First split: training vs. (validation + test)
        val_test_size = val_size + test_size
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, 
            test_size=val_test_size,
            random_state=random_state,
            stratify=stratify_data
        )
        
        # Second split: validation vs. test
        # Adjust test_size to get the correct proportion
        if val_test_size > 0:
            adjusted_test_size = test_size / val_test_size
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test, 
                test_size=adjusted_test_size,
                random_state=random_state,
                stratify=y_val_test if stratify_data is not None else None
            )
        else:
            X_val, X_test, y_val, y_test = X_val_test, pd.DataFrame(), y_val_test, pd.Series()
        
        print(f"Data split: train={len(X_train)} rows ({(1-val_test_size)*100:.1f}%), " + 
              f"validation={len(X_val)} rows ({val_size*100:.1f}%), " + 
              f"test={len(X_test)} rows ({test_size*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Simple train-test split
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_data
        )
        
        print(f"Data split: train={len(X_train)} rows ({(1-test_size)*100:.1f}%), " + 
              f"test={len(X_test)} rows ({test_size*100:.1f}%)")
        
        return X_train, None, X_test, y_train, None, y_test

def visualize_preprocessing(df_original, df_processed, columns=None, n_rows=3, n_cols=2):
    """
    Visualize the effect of preprocessing on the data.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Original dataframe before preprocessing
    df_processed : pandas.DataFrame
        Processed dataframe after preprocessing
    columns : list or None
        List of columns to visualize. If None, selects a subset of numerical columns.
    n_rows, n_cols : int
        Number of rows and columns for the subplot grid
    """
    # If columns not specified, select a subset of numerical columns
    if columns is None:
        common_columns = set(df_original.columns) & set(df_processed.columns)
        columns = list(df_original.select_dtypes(include=[np.number]).columns)
        columns = [col for col in columns if col in common_columns]
        # Limit to n_rows * n_cols plots
        columns = columns[:n_rows*n_cols]
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows*n_cols > 1 else [axes]
    
    for i, col in enumerate(columns):
        if i < len(axes):
            if col in df_original.columns and col in df_processed.columns:
                # Plot histograms for original and processed data
                axes[i].hist(df_original[col].dropna(), alpha=0.5, bins=30, label='Original')
                axes[i].hist(df_processed[col].dropna(), alpha=0.5, bins=30, label='Processed')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, f"Column '{col}' not found in both dataframes", 
                            horizontalalignment='center', verticalalignment='center')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_preprocessing_pipeline(df, config=None):
    """
    Create a preprocessing pipeline based on configuration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to preprocess
    config : dict or None
        Configuration dictionary with preprocessing steps and parameters.
        If None, uses default configuration.
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    # Default configuration
    default_config = {
        'missing_values': {
            'apply': True,
            'strategy': 'mean',
            'columns': None,
        },
        'outliers': {
            'apply': True,
            'method': 'iqr',
            'treatment': 'clip',
            'columns': None,
        },
        'feature_engineering': {
            'apply': True,
            'time_features': True,
        },
        'scaling': {
            'apply': True,
            'method': 'standard',
            'columns': None,
        },
        'categorical_encoding': {
            'apply': True,
            'method': 'one_hot',
            'columns': None,
        }
    }
    
    # Use default config if none provided
    if config is None:
        config = default_config
    else:
        # Update default config with provided config
        for key in default_config:
            if key in config:
                default_config[key].update(config[key])
        config = default_config
    
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Step 1: Handle missing values
    if config['missing_values']['apply']:
        df_processed = handle_missing_values(
            df_processed,
            strategy=config['missing_values']['strategy'],
            columns=config['missing_values']['columns']
        )
    
    # Step 2: Handle outliers
    if config['outliers']['apply']:
        df_processed, outlier_summary = detect_and_handle_outliers(
            df_processed,
            method=config['outliers']['method'],
            treatment=config['outliers']['treatment'],
            columns=config['outliers']['columns']
        )
    
    # Step 3: Feature engineering
    if config['feature_engineering']['apply']:
        if config['feature_engineering']['time_features']:
            df_processed = engineer_time_features(df_processed)
    
    # Step 4: Encode categorical variables
    if config['categorical_encoding']['apply']:
        df_processed = encode_categorical_variables(
            df_processed,
            method=config['categorical_encoding']['method'],
            columns=config['categorical_encoding']['columns']
        )
    
    # Step 5: Scale features
    if config['scaling']['apply']:
        df_processed, _ = scale_features(
            df_processed,
            method=config['scaling']['method'],
            columns=config['scaling']['columns']
        )
    
    return df_processed

if __name__ == "__main__":
    # Example usage
    import os
    
    # Check if data file exists, if not generate it
    data_file = "data/social_media_data.csv"
    if not os.path.exists(data_file):
        # Import from data_generation module and generate data
        from data_generation import generate_social_media_data
        df = generate_social_media_data(
            n_samples=1000, 
            output_file=data_file, 
            random_seed=42,
            missing_data_pct=0.05
        )
    else:
        # Load the data
        df = pd.read_csv(data_file)
    
    print("Original data shape:", df.shape)
    print("Missing values count:\n", df.isnull().sum())
    
    # Example preprocessing pipeline
    processed_df = create_preprocessing_pipeline(df)
    
    print("\nProcessed data shape:", processed_df.shape)
    print("Missing values count after preprocessing:\n", processed_df.isnull().sum())
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        processed_df, 
        target_column='engagement_rate',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Visualize the effect of preprocessing
    visualize_preprocessing(df, processed_df, columns=['post_length', 'followers_count'])
