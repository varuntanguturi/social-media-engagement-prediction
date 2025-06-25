import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os

def generate_descriptive_statistics(df, include_categorical=True):
    """
    Generate descriptive statistics for the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    include_categorical : bool
        Whether to include statistics for categorical columns
        
    Returns:
    --------
    dict
        Dictionary containing various descriptive statistics
    """
    stats = {}
    
    # Basic dataframe info
    stats['shape'] = df.shape
    stats['columns'] = df.columns.tolist()
    
    # Numerical statistics
    numeric_df = df.select_dtypes(include=[np.number])
    stats['numeric_columns'] = numeric_df.columns.tolist()
    stats['numeric_summary'] = numeric_df.describe()
    
    # Check for missing values
    stats['missing_values'] = df.isnull().sum()
    stats['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
    
    # Categorical statistics if requested
    if include_categorical:
        cat_df = df.select_dtypes(exclude=[np.number])
        stats['categorical_columns'] = cat_df.columns.tolist()
        
        # For each categorical column, get value counts
        cat_stats = {}
        for col in stats['categorical_columns']:
            cat_stats[col] = df[col].value_counts()
        
        stats['categorical_summary'] = cat_stats
    
    return stats

def plot_feature_distributions(df, columns=None, n_rows=None, n_cols=3, figsize=(15, 12)):
    """
    Plot distribution histograms for numerical features in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to visualize
    columns : list or None
        List of columns to visualize. If None, uses all numerical columns.
    n_rows : int or None
        Number of rows in the subplot grid. If None, calculated based on columns.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        Figure size (width, height) in inches.
    """
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate number of rows needed
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows*n_cols > 1 else [axes]
    
    # Plot each feature distribution
    for i, col in enumerate(columns):
        if i < len(axes):
            # Create histogram with KDE
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_ylabel('Frequency')
            
            # Add vertical line for mean and median
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i].axvline(df[col].median(), color='green', linestyle='-.', label='Median')
            axes[i].legend()
    
    # Hide unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_correlation_heatmap(df, columns=None, method='pearson', figsize=(12, 10), cmap='coolwarm',
                            annotate=True, mask_upper=False):
    """
    Plot correlation heatmap to show relationships between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    columns : list or None
        List of columns to include. If None, uses all numerical columns.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    figsize : tuple
        Figure size (width, height) in inches
    cmap : str
        Colormap to use for the heatmap
    annotate : bool
        Whether to annotate the heatmap with correlation values
    mask_upper : bool
        Whether to mask the upper triangle of the heatmap
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Select columns for correlation
    if columns is None:
        # Use all numeric columns
        corr_df = df.select_dtypes(include=[np.number])
    else:
        corr_df = df[columns]
    
    # Calculate the correlation matrix
    corr_matrix = corr_df.corr(method=method)
    
    # Create a mask for the upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=annotate, 
        mask=mask,
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        linewidths=0.5, 
        fmt='.2f'
    )
    
    plt.title(f'Feature Correlation Heatmap ({method.capitalize()})')
    plt.tight_layout()
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_target_correlations(df, target_column, top_n=None, figsize=(10, 6)):
    """
    Plot the correlation between features and the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    target_column : str
        The name of the target column
    top_n : int or None
        If specified, shows only the top N features by correlation magnitude.
        If None, shows all features.
    figsize : tuple
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Calculate correlation of all numeric features with the target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ensure target column is included
    if target_column not in numeric_cols:
        print(f"Target column '{target_column}' is not numeric or not in the dataframe")
        return None
    
    # Calculate correlations with target
    correlations = df[numeric_cols].corr()[target_column].sort_values(ascending=False)
    
    # Drop the target's correlation with itself
    correlations = correlations.drop(target_column)
    
    # Take top N by absolute correlation if specified
    if top_n is not None:
        correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
        correlations = correlations.iloc[:top_n]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create horizontal bar chart
    ax = sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(correlations.values):
        ax.text(v + 0.01 if v >= 0 else v - 0.08, i, f'{v:.2f}', va='center')
    
    plt.title(f'Feature Correlations with {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_scatter_with_target(df, feature_column, target_column, figsize=(10, 6), add_regression=True):
    """
    Create a scatter plot of a feature versus the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to visualize
    feature_column : str
        The name of the feature column to plot
    target_column : str
        The name of the target column
    figsize : tuple
        Figure size (width, height) in inches
    add_regression : bool
        Whether to add a regression line to the scatter plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot with regression line
    if add_regression:
        ax = sns.regplot(
            x=feature_column,
            y=target_column,
            data=df,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
    else:
        ax = sns.scatterplot(
            x=feature_column,
            y=target_column,
            data=df,
            alpha=0.5
        )
    
    plt.title(f'{feature_column} vs. {target_column}')
    plt.xlabel(feature_column)
    plt.ylabel(target_column)
    plt.tight_layout()
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_time_based_analysis(df, time_column, target_column, time_unit='hour', 
                             figsize=(12, 6), plot_type='line', error_bars=True):
    """
    Analyze and visualize how the target variable changes over time units.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    time_column : str
        Column containing time information (hour, day, etc.)
    target_column : str
        The target variable to analyze
    time_unit : str
        The time unit for aggregation ('hour', 'day', 'month', etc.)
    figsize : tuple
        Figure size (width, height) in inches
    plot_type : str
        Type of plot to create: 'line' or 'bar'
    error_bars : bool
        Whether to add error bars showing standard deviation
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    # Group by time unit and calculate mean and std of target
    if time_unit == 'hour' and time_column in df.columns:
        # Round to nearest hour if needed
        if df[time_column].dtype == 'float':
            time_groups = df.copy()
            time_groups['hour_group'] = np.floor(df[time_column]).astype(int)
            grouped = time_groups.groupby('hour_group')[target_column].agg(['mean', 'std'])
            grouped.index = grouped.index.astype(int)
        else:
            grouped = df.groupby(time_column)[target_column].agg(['mean', 'std'])
    else:
        # For other time units, assume column already contains appropriate category
        grouped = df.groupby(time_column)[target_column].agg(['mean', 'std'])
    
    # Create the plot
    if plot_type == 'line':
        ax = plt.plot(grouped.index, grouped['mean'], marker='o', linestyle='-', color='blue')
        
        # Add error bars if requested
        if error_bars:
            plt.fill_between(grouped.index, 
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2, color='blue')
    else:
        # Bar plot
        ax = plt.bar(grouped.index, grouped['mean'], yerr=grouped['std'] if error_bars else None,
                     alpha=0.7, capsize=5)
    
    # Formatting for hour of day
    if time_unit == 'hour':
        plt.xticks(range(0, 24))
        plt.xlabel('Hour of Day')
    else:
        plt.xlabel(time_column)
    
    plt.ylabel(f'Average {target_column}')
    plt.title(f'{target_column} by {time_unit.capitalize()}')
    plt.tight_layout()
    
    # If the target column is a percentage, format y-axis
    if 'rate' in target_column.lower() or 'percent' in target_column.lower():
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_feature_pairplot(df, columns=None, target_column=None, diag_kind='kde', 
                          plot_kind='scatter', height=2.5, hue=None):
    """
    Create a pairplot showing relationships between multiple features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to visualize
    columns : list or None
        List of columns to include. If None, uses a selection of numerical columns.
    target_column : str or None
        Target column for coloring points. If None, no color differentiation.
    diag_kind : str
        Kind of plot for the diagonal: 'hist', 'kde'
    plot_kind : str
        Kind of plot for the off-diagonal: 'scatter', 'reg'
    height : float
        Height (in inches) of each facet
    hue : str or None
        Variable for color encoding. If None and target_column is provided, uses target_column.
        
    Returns:
    --------
    seaborn.PairGrid
        The created pairplot object
    """
    # If columns not specified, select a subset of numerical columns to avoid too many plots
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Limit to 5-6 columns for readability
        if len(numeric_cols) > 6:
            corr_with_target = df[numeric_cols].corr()[target_column] if target_column else None
            if corr_with_target is not None:
                # Use columns with highest absolute correlation
                columns = corr_with_target.abs().nlargest(5).index.tolist()
            else:
                columns = numeric_cols[:5]
        else:
            columns = numeric_cols
    
    # Create a dataframe with only selected columns
    plot_df = df[columns].copy()
    
    # Add target column for coloring if specified
    if target_column and target_column not in columns and target_column in df.columns:
        plot_df[target_column] = df[target_column]
        hue = target_column
    
    # Create the pairplot
    pairplot = sns.pairplot(
        plot_df,
        diag_kind=diag_kind,
        kind=plot_kind,
        height=height,
        hue=hue,
        plot_kws={'alpha': 0.6}
    )
    
    pairplot.fig.suptitle('Feature Pair Relationships', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return pairplot

def plot_categorical_analysis(df, cat_column, target_column, figsize=(10, 6), plot_type='box'):
    """
    Analyze how the target variable varies across different categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    cat_column : str
        The categorical column to group by
    target_column : str
        The target variable to analyze
    figsize : tuple
        Figure size (width, height) in inches
    plot_type : str
        Type of plot: 'box', 'violin', or 'bar'
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'box':
        # Box plot showing distribution of target for each category
        sns.boxplot(x=cat_column, y=target_column, data=df)
        plt.title(f'Distribution of {target_column} by {cat_column}')
        
    elif plot_type == 'violin':
        # Violin plot for more detailed distribution
        sns.violinplot(x=cat_column, y=target_column, data=df, inner='quartile')
        plt.title(f'Detailed Distribution of {target_column} by {cat_column}')
        
    elif plot_type == 'bar':
        # Bar plot showing mean of target for each category
        sns.barplot(x=cat_column, y=target_column, data=df, ci=95)
        plt.title(f'Average {target_column} by {cat_column} (with 95% CI)')
    
    plt.xlabel(cat_column)
    plt.ylabel(target_column)
    plt.tight_layout()
    plt.xticks(rotation=45 if len(df[cat_column].unique()) > 5 else 0)
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def plot_outlier_analysis(df, columns=None, n_rows=None, n_cols=2, figsize=(12, 15)):
    """
    Create box plots to visualize outliers in numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    columns : list or None
        List of columns to analyze. If None, uses all numeric columns.
    n_rows : int or None
        Number of rows in the subplot grid. If None, calculated based on columns.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate number of rows needed
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy iteration
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create box plots for each feature
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f'Box Plot of {col}')
            axes[i].set_ylabel(col)
    
    # Hide unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_feature_importance_plot(importance_df, title="Feature Importance", figsize=(10, 8)):
    """
    Create a horizontal bar chart of feature importances.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame or Series
        DataFrame or Series with feature names as index and importance values
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    # If passed a DataFrame, ensure it has right structure
    if isinstance(importance_df, pd.DataFrame):
        if 'importance' in importance_df.columns and 'feature' in importance_df.columns:
            # Already in the right format
            plot_df = importance_df
        else:
            # Try to convert from other format
            plot_df = pd.DataFrame({
                'feature': importance_df.index,
                'importance': importance_df.iloc[:, 0]
            })
    else:
        # Convert Series to DataFrame
        plot_df = pd.DataFrame({
            'feature': importance_df.index,
            'importance': importance_df.values
        })
    
    # Sort by importance
    plot_df = plot_df.sort_values('importance', ascending=False)
    
    # Create the plot
    ax = sns.barplot(x='importance', y='feature', data=plot_df, palette='viridis')
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    fig = plt.gcf()
    plt.show()
    
    return fig

def save_eda_plots(df, target_column, output_dir='data/eda_plots', prefix='eda', 
                   format='png', dpi=300):
    """
    Run a standard EDA workflow and save all plots to a directory.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    target_column : str
        Name of the target column
    output_dir : str
        Directory to save plots
    prefix : str
        Prefix for all filenames
    format : str
        Image format (png, jpg, pdf, etc.)
    dpi : int
        Resolution for saved images
        
    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate descriptive statistics
    stats = generate_descriptive_statistics(df)
    
    # 1. Feature distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig = plot_feature_distributions(df, columns=numeric_cols)
    fig.savefig(f"{output_dir}/{prefix}_distributions.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Correlation heatmap
    fig = plot_correlation_heatmap(df)
    fig.savefig(f"{output_dir}/{prefix}_correlation_heatmap.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Target correlations
    fig = plot_target_correlations(df, target_column)
    fig.savefig(f"{output_dir}/{prefix}_target_correlations.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Scatter plots for top correlated features
    corr = df.corr()[target_column].abs().sort_values(ascending=False)
    top_features = corr[corr.index != target_column].head(3).index
    
    for feature in top_features:
        fig = plot_scatter_with_target(df, feature, target_column)
        fig.savefig(f"{output_dir}/{prefix}_scatter_{feature}_{target_column}.{format}", 
                   dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    # 5. Time-based analysis if applicable
    time_columns = [col for col in df.columns if 'time' in col.lower() or 
                   'hour' in col.lower() or 'day' in col.lower()]
    
    for time_col in time_columns:
        try:
            fig = plot_time_based_analysis(df, time_col, target_column)
            fig.savefig(f"{output_dir}/{prefix}_time_analysis_{time_col}.{format}", 
                       dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        except:
            print(f"Could not create time analysis for {time_col}")
    
    # 6. Pairplot for key features
    pair = plot_feature_pairplot(df, target_column=target_column)
    pair.savefig(f"{output_dir}/{prefix}_pairplot.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(pair.fig)
    
    # 7. Outlier analysis
    fig = plot_outlier_analysis(df, columns=numeric_cols[:6])  # Limit to 6 for readability
    fig.savefig(f"{output_dir}/{prefix}_outliers.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All EDA plots saved to {output_dir}")

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
            random_seed=42
        )
    else:
        # Load the data
        df = pd.read_csv(data_file)
    
    # Apply preprocessing if needed
    from data_preprocessing import handle_missing_values
    df = handle_missing_values(df)
    
    # Run exploratory analysis
    print("Running exploratory analysis...")
    
    # Generate and print descriptive statistics
    stats = generate_descriptive_statistics(df)
    print("\nDataset Shape:", stats['shape'])
    print("\nNumerical Summary:")
    print(stats['numeric_summary'])
    
    # Create some example plots
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_correlations(df, 'engagement_rate', top_n=10)
    
    # If time columns exist
    if 'post_hour' in df.columns:
        plot_time_based_analysis(df, 'post_hour', 'engagement_rate')
    
    # Feature pair relationships
    plot_feature_pairplot(df, target_column='engagement_rate')
    
    # Save all plots to a directory
    save_eda_plots(df, 'engagement_rate')
