import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

def train_linear_regression(X_train, y_train, **kwargs):
    """
    Train a Linear Regression model.
    
    Linear Regression is a simple model that assumes a linear relationship between 
    the input features and the target variable. It's a good baseline model and works 
    well when the relationship in the data is approximately linear.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    **kwargs : additional parameters
        Additional parameters to pass to the LinearRegression constructor
    
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained linear regression model
    """
    # Create the model with specified parameters or defaults
    model = LinearRegression(**kwargs)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Print model information
    print("Linear Regression Model:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    return model

def train_regularized_linear_models(X_train, y_train, model_type='ridge', **kwargs):
    """
    Train a regularized linear regression model (Ridge, Lasso, or ElasticNet).
    
    Regularized linear models add penalties to the standard linear regression 
    to prevent overfitting, especially when dealing with many features or 
    highly correlated features.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    model_type : str
        Type of model: 'ridge', 'lasso', or 'elasticnet'
    **kwargs : additional parameters
        Additional parameters to pass to the model constructor
    
    Returns:
    --------
    sklearn.base.BaseEstimator
        Trained regularized linear model
    """
    # Select the model type
    if model_type.lower() == 'ridge':
        # Ridge regression: L2 regularization (penalizes sum of squared coefficients)
        # Good when many small/medium sized effects
        model = Ridge(**kwargs)
        model_name = "Ridge Regression"
    elif model_type.lower() == 'lasso':
        # Lasso regression: L1 regularization (penalizes sum of absolute coefficients)
        # Good for feature selection as it can zero out unimportant features
        model = Lasso(**kwargs)
        model_name = "Lasso Regression"
    elif model_type.lower() == 'elasticnet':
        # ElasticNet: Combines L1 and L2 regularization
        # Good when many correlated features
        model = ElasticNet(**kwargs)
        model_name = "ElasticNet Regression"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'ridge', 'lasso', or 'elasticnet'.")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Print model information
    print(f"{model_name} Model:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    return model

def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest Regressor model.
    
    Random Forest is an ensemble method that builds multiple decision trees and 
    merges their predictions. It works well with non-linear relationships, handles 
    outliers effectively, and is less prone to overfitting compared to a single decision tree.
    
    For social media engagement prediction, Random Forest can capture complex interactions
    between features like post timing, content length, and follower count.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    **kwargs : additional parameters
        Additional parameters to pass to RandomForestRegressor constructor
        Key parameters include:
        - n_estimators: number of trees (higher = more robust, but slower)
        - max_depth: maximum depth of trees (control model complexity)
        - min_samples_split: min samples required to split a node (prevent overfitting)
    
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    # Default parameters if not specified
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
    }
    
    # Update defaults with any provided parameters
    params = {**default_params, **kwargs}
    
    # Create and train the model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Random Forest Regressor Model:")
    print(f"Number of Trees: {model.n_estimators}")
    print(f"Feature Importances: {model.feature_importances_}")
    
    return model

def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting Regressor model.
    
    Gradient Boosting builds an ensemble of decision trees sequentially, with each tree 
    correcting the errors of the previous ones. This model is often more accurate than 
    Random Forest but may be more prone to overfitting if not properly tuned.
    
    For social media engagement prediction, Gradient Boosting is well-suited to capture 
    subtle patterns in user engagement based on content and timing features.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    **kwargs : additional parameters
        Additional parameters to pass to GradientBoostingRegressor constructor
        Key parameters include:
        - n_estimators: number of boosting stages (trees)
        - learning_rate: contribution of each tree to the final solution
        - max_depth: maximum depth of trees
        - subsample: fraction of samples for fitting individual trees
    
    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    """
    # Default parameters if not specified
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 1.0,
        'random_state': 42
    }
    
    # Update defaults with any provided parameters
    params = {**default_params, **kwargs}
    
    # Create and train the model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Gradient Boosting Regressor Model:")
    print(f"Number of Trees: {model.n_estimators}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Feature Importances: {model.feature_importances_}")
    
    return model

def train_svr(X_train, y_train, scale_data=True, **kwargs):
    """
    Train a Support Vector Regression model.
    
    SVR uses the same principles as SVM for classification, but for regression tasks.
    It's effective for data with non-linear relationships but can be slow on larger datasets.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    scale_data : bool
        Whether to scale the data before training (recommended for SVR)
    **kwargs : additional parameters
        Additional parameters to pass to SVR constructor
    
    Returns:
    --------
    sklearn.pipeline.Pipeline or sklearn.svm.SVR
        Trained SVR model (within pipeline if scaling is applied)
    """
    # Default parameters if not specified
    default_params = {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    }
    
    # Update defaults with any provided parameters
    params = {**default_params, **kwargs}
    
    # SVR works best with scaled data
    if scale_data:
        # Create a pipeline with scaling and SVR
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**params))
        ])
    else:
        model = SVR(**params)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    print("Support Vector Regression Model:")
    print(f"Kernel: {params['kernel']}")
    print(f"C: {params['C']}")
    print(f"Epsilon: {params['epsilon']}")
    
    return model

def tune_hyperparameters(X_train, y_train, model_type='random_forest', cv=5, 
                         scoring='neg_mean_squared_error', search_method='grid', n_iter=10):
    """
    Perform hyperparameter tuning for a specified model type.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Target variable
    model_type : str
        Type of model to tune: 'random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso'
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric to optimize
    search_method : str
        'grid' for GridSearchCV or 'random' for RandomizedSearchCV
    n_iter : int
        Number of iterations for RandomizedSearchCV
        
    Returns:
    --------
    dict
        Dictionary containing the best estimator, best parameters, and CV results
    """
    # Set up parameter grids for different model types
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
    elif model_type == 'linear':
        model = LinearRegression()
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'copy_X': [True]
        }
        
    elif model_type == 'ridge':
        model = Ridge(random_state=42)
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
        
    elif model_type == 'lasso':
        model = Lasso(random_state=42)
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 5000]
        }
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform hyperparameter search
    start_time = time.time()
    
    if search_method == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
    else:  # random search
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    search.fit(X_train, y_train)
    
    # Summarize results
    print(f"\nHyperparameter Tuning for {model_type.title()} (using {search_method} search):")
    print(f"Best Score ({scoring}): {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Return results
    results = {
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }
    
    return results

def perform_cross_validation(model, X, y, cv=5, scoring=['neg_mean_squared_error', 'r2']):
    """
    Perform cross-validation for model evaluation.
    
    Parameters:
    -----------
    model : sklearn.base.BaseEstimator
        Model to evaluate
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : pandas.Series or numpy.ndarray
        Target variable
    cv : int or sklearn.model_selection._split.BaseCrossValidator
        Number of folds or cross-validator object
    scoring : str or list
        Scoring metric(s) to use
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation scores
    """
    cv_results = {}
    
    # Make scoring a list if it's a string
    if isinstance(scoring, str):
        scoring = [scoring]
    
    # Perform cross-validation for each scoring metric
    for score in scoring:
        start_time = time.time()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=score, n_jobs=-1)
        time_taken = time.time() - start_time
        
        # Store results
        cv_results[score] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist(),
            'time': time_taken
        }
        
        # Convert negative MSE to RMSE for easier interpretation
        if score == 'neg_mean_squared_error':
            rmse_scores = np.sqrt(-cv_scores)
            cv_results['rmse'] = {
                'mean': rmse_scores.mean(),
                'std': rmse_scores.std(),
                'scores': rmse_scores.tolist()
            }
            print(f"RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
        else:
            print(f"{score}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    print(f"Cross-validation completed in {time_taken:.2f} seconds")
    
    return cv_results

def save_model(model, model_name=None, directory='models'):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model to save
    model_name : str or None
        Name for the model file. If None, generates a name based on model type and timestamp.
    directory : str
        Directory to save model in
        
    Returns:
    --------
    str
        Path to the saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate filename if not provided
    if model_name is None:
        # Extract model type
        model_type = type(model).__name__
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_{timestamp}.pkl"
    elif not model_name.endswith('.pkl'):
        model_name += '.pkl'
    
    # Full path to save the model
    model_path = os.path.join(directory, model_name)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    object
        Loaded model
    """
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model

def compare_models(models, X_test, y_test):
    """
    Compare multiple models on test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        Test target values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Add to results
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values('RMSE')
    
    return results_df

def get_feature_importances(model, feature_names):
    """
    Extract feature importances from a trained model.
    
    Parameters:
    -----------
    model : trained model
        Model that supports feature importances
    feature_names : list
        Names of features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features and their importance scores
    """
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # For linear models, check for coef_ attribute
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    # For pipeline, check if the final step has feature importances
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
        importances = model.steps[-1][1].feature_importances_
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'coef_'):
        importances = np.abs(model.steps[-1][1].coef_)
    else:
        raise AttributeError("Model does not support feature importances")
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def recommend_model_for_social_media(data_size, feature_count, interpretability_needed=False):
    """
    Recommend a model type based on the social media data characteristics.
    
    Parameters:
    -----------
    data_size : int
        Number of samples in the dataset
    feature_count : int
        Number of features
    interpretability_needed : bool
        Whether model interpretability is important
        
    Returns:
    --------
    str
        Recommended model type with explanation
    """
    recommendation = "Model Recommendation for Social Media Engagement Prediction:\n\n"
    
    # Decision logic based on data characteristics
    if data_size < 1000:
        if interpretability_needed:
            model = "Linear Regression or Ridge Regression"
            explanation = ("For smaller datasets with fewer than 1000 samples, simpler models like "
                          "linear regression or ridge regression are recommended to avoid overfitting. "
                          "These models also provide interpretable coefficients that explain how "
                          "each feature affects engagement rates.")
        else:
            model = "Random Forest Regressor"
            explanation = ("For smaller datasets where interpretability is not crucial, Random Forest "
                          "is recommended as it handles non-linear relationships well and is less prone "
                          "to overfitting compared to more complex models.")
    
    elif data_size < 10000:
        if feature_count > 20 or not interpretability_needed:
            model = "Gradient Boosting Regressor"
            explanation = ("For medium-sized datasets with many features, Gradient Boosting often "
                          "provides the best performance by iteratively correcting errors. It can "
                          "capture complex patterns in social media engagement data like time-based trends "
                          "and interaction effects between content features.")
        else:
            model = "Random Forest or ElasticNet"
            explanation = ("For medium-sized datasets where some interpretability is desired, "
                          "Random Forest offers a good balance of performance and partial interpretability "
                          "through feature importance. ElasticNet is useful if there are many correlated "
                          "features like different content metrics.")
    
    else:  # Large dataset
        if interpretability_needed:
            model = "Gradient Boosting with feature importance analysis"
            explanation = ("For large datasets requiring interpretability, Gradient Boosting with "
                          "post-hoc feature importance analysis offers a good balance. While not as "
                          "directly interpretable as linear models, feature importance scores help "
                          "identify key drivers of engagement.")
        else:
            model = "Ensemble of Gradient Boosting and Neural Networks"
            explanation = ("For large social media datasets where maximum predictive power is the goal, "
                          "an ensemble approach combining Gradient Boosting and Neural Networks often "
                          "yields the best results, capturing both structured patterns and complex "
                          "non-linear relationships in engagement behavior.")
    
    # Social media specific considerations
    social_media_notes = (
        "\nSpecial considerations for social media engagement prediction:\n"
        "1. Temporal patterns are important - ensure models can capture time-of-day and day-of-week effects\n"
        "2. Interaction terms may improve performance (e.g., post_length × follower_count)\n"
        "3. Consider separate models for different types of content if engagement patterns vary significantly\n"
        "4. For viral prediction, consider specialized approaches like zero-inflated models to handle skewed distributions"
    )
    
    return recommendation + f"Recommended model: {model}\n\n{explanation}\n{social_media_notes}"

if __name__ == "__main__":
    # Example usage of the module
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    print("Generating sample data for demonstration...")
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sample feature names for the demonstration
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    print("\n--- Training Linear Regression Model ---")
    linear_model = train_linear_regression(X_train, y_train)
    
    print("\n--- Training Random Forest Model ---")
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)
    
    print("\n--- Training Gradient Boosting Model ---")
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1)
    
    print("\n--- Model Comparison ---")
    models = {
        "Linear Regression": linear_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model
    }
    comparison = compare_models(models, X_test, y_test)
    print(comparison)
    
    print("\n--- Feature Importances (Random Forest) ---")
    importances = get_feature_importances(rf_model, feature_names)
    print(importances)
    
    print("\n--- Model Recommendation for Social Media Data ---")
    print(recommend_model_for_social_media(data_size=5000, feature_count=15))
