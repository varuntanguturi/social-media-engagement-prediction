import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import os

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate common regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary containing various regression metrics
        
    Notes:
    ------
    - R² (Coefficient of Determination): Measures the proportion of variance in the dependent 
      variable that is predictable from the independent variables. Range: (-∞, 1]
      R² = 1 indicates a perfect fit, R² = 0 indicates the model is no better than a horizontal line.
      
    - MSE (Mean Squared Error): Average of squared differences between predicted and actual values.
      Lower values indicate better fit. Units are squared units of the target variable.
      
    - RMSE (Root Mean Squared Error): Square root of MSE, which brings the units back to 
      those of the target variable. More interpretable than MSE.
      
    - MAE (Mean Absolute Error): Average of absolute differences between predicted and actual values.
      Less sensitive to outliers than MSE/RMSE. Units are the same as the target variable.
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate adjusted R² if possible (need to know number of samples and features)
    # adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Additional metrics for comprehensive evaluation
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-7, None))) * 100
    
    # Create metrics dictionary
    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return metrics

def print_metrics_report(metrics, model_name=None):
    """
    Print a formatted report of regression metrics with interpretations.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing regression metrics from calculate_regression_metrics()
    model_name : str or None
        Name of the model being evaluated
    """
    title = f"Evaluation Metrics for {model_name}" if model_name else "Regression Metrics Report"
    
    print(f"\n{'-' * len(title)}")
    print(title)
    print(f"{'-' * len(title)}")
    
    # Print metrics with interpretations
    print(f"R² Score: {metrics['r2']:.4f}")
    if metrics['r2'] > 0.9:
        print("   Interpretation: Excellent fit! The model explains more than 90% of the variance.")
    elif metrics['r2'] > 0.7:
        print("   Interpretation: Good fit. The model explains a substantial portion of the variance.")
    elif metrics['r2'] > 0.5:
        print("   Interpretation: Moderate fit. The model explains about half of the variance.")
    else:
        print("   Interpretation: Poor fit. Consider trying different features or models.")
    
    print(f"RMSE: {metrics['rmse']:.4f}")
    print("   Interpretation: The average prediction error in the same units as the target variable.")
    
    print(f"MAE: {metrics['mae']:.4f}")
    print("   Interpretation: The average absolute difference between predicted and actual values.")
    
    print(f"MAPE: {metrics['mape']:.2f}%")
    print("   Interpretation: The average percentage error of predictions.")
    
    print(f"{'-' * len(title)}")

def plot_actual_vs_predicted(y_true, y_pred, model_name=None, figsize=(10, 6)):
    """
    Create a scatter plot comparing actual vs. predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model_name : str or None
        Name of the model used for predictions
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Notes:
    ------
    - Perfect predictions would fall on the diagonal line y = x
    - Points above the line represent underestimates
    - Points below the line represent overestimates
    - The spread of points around the line indicates prediction error magnitude
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=50)
    
    # Get the range for both axes
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    # Add perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Set labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    title = 'Actual vs. Predicted Values'
    if model_name:
        title += f' ({model_name})'
    plt.title(title)
    
    # Add metrics to plot
    metrics = calculate_regression_metrics(y_true, y_pred)
    r2, rmse = metrics['r2'], metrics['rmse']
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_residuals(y_true, y_pred, model_name=None, figsize=(12, 8)):
    """
    Create a comprehensive residual analysis plot with multiple subplots.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model_name : str or None
        Name of the model used for predictions
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Notes:
    ------
    - Residuals should ideally:
      1. Be centered around zero (no bias)
      2. Show no patterns when plotted against predicted values (homoscedasticity)
      3. Be normally distributed
      4. Show no autocorrelation (not checked in this function)
    
    - Patterns in residuals can indicate:
      - Heteroscedasticity: Funnel shape indicates inconsistent error variance
      - Missing predictors: Clear patterns suggest missing important variables
      - Non-linearity: Curved patterns suggest linear model may be inappropriate
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Residual Analysis{" for " + model_name if model_name else ""}', 
                fontsize=16, y=1.05)
    
    # Plot 1: Residuals vs. Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals vs. Predicted Values')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of Residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q Plot of Residuals
    from scipy import stats
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Check for Normality)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Absolute Residuals vs. Predicted (check for heteroscedasticity)
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6, edgecolor='k')
    axes[1, 1].set_title('|Residuals| vs. Predicted Values')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Add residual statistics as text
    stats_text = (f"Residual Statistics:\n"
                 f"Mean: {mean_residual:.4f}\n"
                 f"Std Dev: {std_residual:.4f}\n"
                 f"Min: {np.min(residuals):.4f}\n"
                 f"Max: {np.max(residuals):.4f}")
    
    fig.text(0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.show()
    return fig

def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 8)):
    """
    Visualize feature importances from a tree-based model.
    
    Parameters:
    -----------
    model : trained model
        Tree-based model with feature_importances_ attribute (e.g., RandomForest, GradientBoosting)
    feature_names : list
        Names of features used in the model
    top_n : int or None
        If specified, shows only the top N most important features
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Notes:
    ------
    - Feature importance indicates how useful each feature was for making decisions in the model
    - Higher values indicate more important features
    - For tree-based models, this typically measures how much each feature reduced impurity
    - Can help identify which variables most strongly influence engagement predictions
    """
    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # For linear models, check for coef_ attribute (take absolute value)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1 and importances.shape[0] == 1:
            importances = importances[0]  # Flatten if 2D with one row
    # For pipeline, check if the final step has feature importances
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
        importances = model.steps[-1][1].feature_importances_
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'coef_'):
        importances = np.abs(model.steps[-1][1].coef_)
        if importances.ndim > 1 and importances.shape[0] == 1:
            importances = importances[0]
    else:
        raise AttributeError("Model does not support feature importances")
    
    # Create DataFrame for easier manipulation and plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Limit to top N if specified
    if top_n is not None and top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot horizontal bar chart
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
    
    # Add value labels to the bars
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Customize plot
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Add note about interpretation
    note = ("Note: Feature importance scores show the relative influence of each feature in the model.\n"
            "Higher values indicate greater impact on predictions.")
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=10, 
                bbox=dict(facecolor='wheat', alpha=0.3))
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                       figsize=(10, 6)):
    """
    Generate a plot of training and cross-validation scores as a function of the training set size.
    
    Parameters:
    -----------
    estimator : object
        A scikit-learn estimator with fit and predict methods
    X : array-like
        Feature data
    y : array-like
        Target data
    cv : int or cross-validation generator
        Cross-validation strategy
    n_jobs : int
        Number of jobs to run in parallel
    train_sizes : array-like
        Points on the training set size to evaluate
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Notes:
    ------
    - Learning curves help diagnose bias and variance issues in models
    - Closely spaced curves with low error: Good fit
    - High training score but low validation score: Overfitting
    - Both curves plateau at a low score: Underfitting
    - Increasing gap between curves with more data: Might need more complex model
    """
    # Calculate learning curve values
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='neg_mean_squared_error')
    
    # Convert negative MSE to RMSE for better interpretability
    train_scores_rmse = np.sqrt(-train_scores)
    test_scores_rmse = np.sqrt(-test_scores)
    
    # Calculate mean and std
    train_scores_mean = np.mean(train_scores_rmse, axis=1)
    train_scores_std = np.std(train_scores_rmse, axis=1)
    test_scores_mean = np.mean(test_scores_rmse, axis=1)
    test_scores_std = np.std(test_scores_rmse, axis=1)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot learning curves
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training RMSE')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation RMSE')
    
    # Plot standard deviation bands
    plt.fill_between(train_sizes, 
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, 
                    alpha=0.1, color='blue')
    plt.fill_between(train_sizes, 
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, 
                    alpha=0.1, color='green')
    
    # Customize plot
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve Analysis')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Analyze curve patterns to provide diagnostics
    gap = test_scores_mean[-1] - train_scores_mean[-1]
    
    # Add diagnostics text based on learning curve patterns
    if test_scores_mean[-1] > 2 * train_scores_mean[-1]:
        diagnosis = "Diagnosis: Likely overfitting. The model performs much better on training data than validation data."
    elif train_scores_mean[-1] > 0.9 * max(train_scores_mean) and test_scores_mean[-1] > 0.9 * max(test_scores_mean):
        diagnosis = "Diagnosis: Likely underfitting. Both curves plateau at high error rates. Consider a more complex model."
    elif gap < 0.2 * test_scores_mean[-1] and test_scores_mean[-1] < 0.7 * max(test_scores_mean):
        diagnosis = "Diagnosis: Good fit. Small gap between curves and decreasing error with more data."
    else:
        diagnosis = "Diagnosis: More data might help reduce the gap between training and validation performance."
    
    plt.figtext(0.5, 0.01, diagnosis, ha='center', fontsize=10, 
                bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def compare_models_plot(models_dict, X, y, metric='rmse', figsize=(12, 8)):
    """
    Create a visual comparison of multiple models using bar plots of evaluation metrics.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X : array-like
        Feature data for prediction
    y : array-like
        True target values
    metric : str
        Primary metric for sorting ('rmse', 'mae', 'r2')
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    pandas.DataFrame
        DataFrame with all calculated metrics
    
    Notes:
    ------
    - Lower values of RMSE and MAE indicate better models
    - Higher values of R² indicate better models
    - Consider using cross-validation for more robust comparisons
    - Look for consistent performance across multiple metrics
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Generate predictions and calculate metrics for each model
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X)
        metrics = calculate_regression_metrics(y, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'MSE': metrics['mse'],
            'MAPE': metrics['mape']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by specified metric
    if metric == 'rmse':
        results_df = results_df.sort_values('RMSE')
    elif metric == 'mae':
        results_df = results_df.sort_values('MAE')
    elif metric == 'r2':
        results_df = results_df.sort_values('R²', ascending=False)
    
    # Plot RMSE comparison
    axes[0].barh(results_df['Model'], results_df['RMSE'], color='salmon')
    axes[0].set_title('RMSE (lower is better)')
    axes[0].set_xlabel('RMSE Value')
    axes[0].invert_yaxis()  # Best model at top
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot MAE comparison
    axes[1].barh(results_df['Model'], results_df['MAE'], color='skyblue')
    axes[1].set_title('MAE (lower is better)')
    axes[1].set_xlabel('MAE Value')
    axes[1].invert_yaxis()  # Best model at top
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot R² comparison
    axes[2].barh(results_df['Model'], results_df['R²'], color='lightgreen')
    axes[2].set_title('R² (higher is better)')
    axes[2].set_xlabel('R² Value')
    axes[2].invert_yaxis()  # Keep same order as other plots
    axes[2].grid(axis='x', alpha=0.3)
    
    # Add value labels to bars
    for i, ax in enumerate(axes):
        metric_col = ['RMSE', 'MAE', 'R²'][i]
        for j, v in enumerate(results_df[metric_col]):
            ax.text(v + (0.01 if i < 2 else -0.08), j, f'{v:.4f}', va='center')
    
    plt.suptitle('Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Add interpretation guidance
    note = ("Interpretation Guide:\n"
            "• RMSE & MAE: Lower values indicate better models\n"
            "• R²: Higher values indicate better models (best is 1.0)\n"
            "• Consider tradeoffs between metrics and model complexity")
    
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=10, 
                bbox=dict(facecolor='wheat', alpha=0.3))
    
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    return fig, results_df

def evaluate_prediction_segments(y_true, y_pred, segments=3, figsize=(12, 6)):
    """
    Evaluate model performance across different segments of the target variable.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    segments : int
        Number of segments to divide the data into
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    pandas.DataFrame
        DataFrame with metrics for each segment
        
    Notes:
    ------
    - Models often perform differently across the range of target values
    - This analysis helps identify where the model performs well or poorly
    - For engagement prediction, knowing if your model is better at predicting
      high or low engagement posts can inform business decisions
    """
    # Create a DataFrame with true and predicted values
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': np.abs(y_true - y_pred),
        'pct_error': np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-7, None)) * 100
    })
    
    # Sort by true values and divide into segments
    df = df.sort_values('y_true').reset_index(drop=True)
    df['segment'] = pd.qcut(df.index, segments, labels=[f'Segment {i+1}' for i in range(segments)])
    
    # Calculate metrics for each segment
    segment_metrics = df.groupby('segment').apply(
        lambda x: pd.Series({
            'min_value': x['y_true'].min(),
            'max_value': x['y_true'].max(),
            'mean_value': x['y_true'].mean(),
            'count': len(x),
            'rmse': np.sqrt(mean_squared_error(x['y_true'], x['y_pred'])),
            'mae': mean_absolute_error(x['y_true'], x['y_pred']),
            'mape': x['pct_error'].mean(),
            'r2': r2_score(x['y_true'], x['y_pred']) if len(x) > 1 else np.nan
        })
    ).reset_index()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot RMSE and MAE by segment
    segment_metrics.plot(x='segment', y=['rmse', 'mae'], kind='bar', ax=ax1)
    ax1.set_title('Error Metrics by Target Segment')
    ax1.set_ylabel('Error Value')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot R² by segment
    segment_metrics.plot(x='segment', y='r2', kind='bar', color='green', ax=ax2)
    ax2.set_title('R² Score by Target Segment')
    ax2.set_ylabel('R² Value')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Performance Across Target Value Segments', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Add segment ranges as text
    segment_info = "Segment Ranges:\n" + "\n".join(
        [f"Segment {i+1}: {row['min_value']:.2f} to {row['max_value']:.2f}" 
         for i, row in segment_metrics.iterrows()]
    )
    plt.figtext(0.01, 0.01, segment_info, fontsize=9, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.show()
    
    return fig, segment_metrics

def save_evaluation_results(model, X_test, y_test, feature_names, model_name=None, output_dir='evaluation_results'):
    """
    Generate and save a comprehensive set of evaluation visualizations for a model.
    
    Parameters:
    -----------
    model : trained model
        Model to evaluate
    X_test : array-like
        Test feature data
    y_test : array-like
        Test target data
    feature_names : list
        Names of features
    model_name : str or None
        Name of the model
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary with paths to saved files and calculated metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        model_name = type(model).__name__
    
    # Create safe filename base
    safe_name = model_name.replace(' ', '_').lower()
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred)
    
    # Create and save plots
    results = {'metrics': metrics, 'files': {}}
    
    # 1. Actual vs Predicted plot
    fig1 = plot_actual_vs_predicted(y_test, y_pred, model_name)
    fig1_path = os.path.join(output_dir, f'{safe_name}_actual_vs_predicted.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    results['files']['actual_vs_predicted'] = fig1_path
    
    # 2. Residual analysis
    fig2 = plot_residuals(y_test, y_pred, model_name)
    fig2_path = os.path.join(output_dir, f'{safe_name}_residuals.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    results['files']['residuals'] = fig2_path
    
    # 3. Feature importance (if supported)
    try:
        fig3 = plot_feature_importance(model, feature_names)
        fig3_path = os.path.join(output_dir, f'{safe_name}_feature_importance.png')
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        results['files']['feature_importance'] = fig3_path
    except (AttributeError, ValueError) as e:
        print(f"Couldn't generate feature importance plot: {e}")
    
    # 4. Learning curve
    try:
        fig4 = plot_learning_curve(model, X_test, y_test)
        fig4_path = os.path.join(output_dir, f'{safe_name}_learning_curve.png')
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        results['files']['learning_curve'] = fig4_path
    except Exception as e:
        print(f"Couldn't generate learning curve: {e}")
    
    # 5. Prediction segments analysis
    fig5, segment_metrics = evaluate_prediction_segments(y_test, y_pred)
    fig5_path = os.path.join(output_dir, f'{safe_name}_prediction_segments.png')
    fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    results['files']['prediction_segments'] = fig5_path
    results['segment_metrics'] = segment_metrics.to_dict()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, f'{safe_name}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    results['files']['metrics_csv'] = metrics_path
    
    print(f"Evaluation metrics and visualizations saved to {output_dir}")