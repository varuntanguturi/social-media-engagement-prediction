import argparse
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Import modules from src directory
from src.data_generation import generate_social_media_data
from src.data_preprocessing import (handle_missing_values, scale_features,
                                   engineer_time_features, detect_and_handle_outliers,
                                   encode_categorical_variables, split_dataset,
                                   create_preprocessing_pipeline)
from src.exploratory_analysis import (generate_descriptive_statistics,
                                     plot_feature_distributions, plot_correlation_heatmap,
                                     plot_target_correlations, plot_scatter_with_target,
                                     plot_time_based_analysis, plot_feature_pairplot,
                                     save_eda_plots)
from src.model_building import (train_linear_regression, train_random_forest,
                              train_gradient_boosting, tune_hyperparameters,
                              perform_cross_validation, save_model, load_model,
                              compare_models, get_feature_importances)
from src.model_evaluation import (calculate_regression_metrics, print_metrics_report,
                                plot_actual_vs_predicted, plot_residuals,
                                plot_feature_importance, plot_learning_curve,
                                compare_models_plot, save_evaluation_results)

# Set up logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    log_file : str or None
        Path to log file. If None, logs only to console.
    """
    # Create logs directory if logging to file
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def generate_data(args):
    """
    Generate synthetic social media data based on command line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    pandas.DataFrame
        Generated data
    """
    logging.info("Starting data generation phase...")
    
    try:
        # Generate data with parameters from command line
        df = generate_social_media_data(
            n_samples=args.samples,
            output_file=args.output if args.save_data else None,
            random_seed=args.seed,
            missing_data_pct=args.missing_pct,
            noise_level=args.noise
        )
        logging.info(f"Successfully generated {len(df)} samples")
        
        # Display basic info about the data
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Features: {', '.join(df.columns)}")
        logging.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error during data generation: {str(e)}")
        raise

def preprocess_data(df, args):
    """
    Preprocess the data according to command line arguments.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data to preprocess
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logging.info("Starting data preprocessing phase...")
    
    try:
        # Create preprocessing configuration from command line arguments
        preproc_config = {
            'missing_values': {
                'apply': not args.skip_imputation,
                'strategy': args.imputation_strategy
            },
            'outliers': {
                'apply': not args.skip_outlier_treatment,
                'method': args.outlier_method,
                'treatment': args.outlier_treatment
            },
            'feature_engineering': {
                'apply': not args.skip_feature_engineering,
                'time_features': True
            },
            'scaling': {
                'apply': not args.skip_scaling,
                'method': args.scaling_method
            },
            'categorical_encoding': {
                'apply': True,
                'method': 'one_hot'
            }
        }
        
        # Log information about categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            logging.info(f"Categorical columns to encode: {', '.join(categorical_cols)}")
            encoding_info = f"Using {preproc_config['categorical_encoding']['method']} encoding"
            test_encoding = encode_categorical_variables(df[categorical_cols].head(1), method='one_hot')
            logging.info(f"{encoding_info} will expand to {test_encoding.shape[1]} features")

        # Apply preprocessing pipeline
        logging.info("Applying preprocessing pipeline...")
        df_processed = create_preprocessing_pipeline(df, preproc_config)
        
        # Log outlier detection results
        if not args.skip_outlier_treatment:
            outlier_count = detect_and_handle_outliers(df, method=args.outlier_method, treatment='flag').sum()
            logging.info(f"Detected {outlier_count} outliers using {args.outlier_method} method")

        # Split the dataset
        logging.info(f"Splitting dataset (test_size={args.test_size}, val_size={args.val_size})...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df_processed,
            target_column='engagement_rate',
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed
        )
        
        logging.info(f"Preprocessing complete. Training set shape: {X_train.shape}")
        
        # Log feature scaling information if scaling is applied
        if not args.skip_scaling:
            logging.info(f"Features scaled using {args.scaling_method} scaling")
            scale_info = scale_features(pd.DataFrame(X_train.iloc[0:1]), method=args.scaling_method, return_scaler=True)[1]
            logging.info(f"Scaling transformation initialized: {scale_info}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

def explore_data(df, args):
    """
    Perform exploratory data analysis on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to analyze
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary with EDA results
    """
    logging.info("Starting exploratory data analysis phase...")
    
    try:
        # Generate basic statistics
        stats = generate_descriptive_statistics(df)
        logging.info("Generated descriptive statistics")
        
        # Create visualizations if not skipped
        if not args.skip_visualization:
            logging.info("Generating visualizations...")
            
            # Save EDA plots to directory if specified
            if args.save_plots:
                output_dir = args.plot_dir if args.plot_dir else 'data/eda_plots'
                logging.info(f"Saving EDA plots to {output_dir}")
                save_eda_plots(df, 'engagement_rate', output_dir=output_dir)
            else:
                # Interactive mode - show some key plots
                plot_feature_distributions(df)
                plot_correlation_heatmap(df)
                plot_target_correlations(df, 'engagement_rate')
                
                # If time columns exist
                if 'post_hour' in df.columns:
                    plot_time_based_analysis(df, 'post_hour', 'engagement_rate')
                
                # Plot pairwise relationships
                plot_feature_pairplot(df, target_column='engagement_rate')
        
        # Check for time features that could be engineered
        if 'post_time' in df.columns or 'timestamp' in df.columns:
            time_col = 'post_time' if 'post_time' in df.columns else 'timestamp'
            logging.info(f"Time features could be engineered from '{time_col}' column")
            sample_time_features = engineer_time_features(df.sample(min(5, len(df))), time_column=time_col)
            logging.info(f"Sample time features that can be created: {', '.join(sample_time_features.columns)}")

        logging.info("Exploratory analysis complete")
        return {'stats': stats}
    
    except Exception as e:
        logging.error(f"Error during exploratory analysis: {str(e)}")
        raise

def train_models(X_train, y_train, X_val=None, y_val=None, args=None):
    """
    Train machine learning models based on command line arguments.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    X_val : pandas.DataFrame or None
        Validation features
    y_val : pandas.Series or None
        Validation target values
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    logging.info("Starting model training phase...")
    
    models = {}
    
    try:
        # Train linear regression if specified
        if 'linear' in args.models:
            logging.info("Training Linear Regression model...")
            linear_model = train_linear_regression(X_train, y_train)
            models['Linear Regression'] = linear_model
        
        # Train random forest if specified
        if 'random_forest' in args.models:
            logging.info("Training Random Forest model...")
            rf_params = {
                'n_estimators': args.rf_trees,
                'max_depth': args.rf_depth,
                'random_state': args.seed
            }
            rf_model = train_random_forest(X_train, y_train, **rf_params)
            models['Random Forest'] = rf_model
        
        # Train gradient boosting if specified
        if 'gradient_boosting' in args.models:
            logging.info("Training Gradient Boosting model...")
            gb_params = {
                'n_estimators': args.gb_trees,
                'learning_rate': args.gb_lr,
                'max_depth': args.gb_depth,
                'random_state': args.seed
            }
            gb_model = train_gradient_boosting(X_train, y_train, **gb_params)
            models['Gradient Boosting'] = gb_model
        
        # Perform hyperparameter tuning if specified
        if args.tune_hyperparams and args.tune_model in args.models:
            logging.info(f"Tuning hyperparameters for {args.tune_model}...")
            tuning_results = tune_hyperparameters(
                X_train, y_train,
                model_type=args.tune_model,
                cv=args.cv_folds,
                search_method='grid' if args.grid_search else 'random',
                n_iter=args.random_iter
            )
            
            # Add tuned model to models dictionary
            models[f"Tuned {args.tune_model.title()}"] = tuning_results['best_estimator']
        
        # Perform cross-validation if specified
        if args.cross_validate:
            for name, model in models.items():
                logging.info(f"Performing cross-validation for {name}...")
                cv_results = perform_cross_validation(
                    model, X_train, y_train,
                    cv=args.cv_folds,
                    scoring=['neg_mean_squared_error', 'r2']
                )
                # Use cv_results to log cross-validation performance
                logging.info(f"Cross-validation results for {name}: MSE = {-cv_results['test_neg_mean_squared_error'].mean():.4f}, R² = {cv_results['test_r2'].mean():.4f}")
        
        # Save models if specified
        if args.save_models:
            model_dir = args.model_dir if args.model_dir else 'models'
            for name, model in models.items():
                model_filename = f"{name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                save_model(model, model_filename, directory=model_dir)
        
        logging.info(f"Successfully trained {len(models)} models")
        return models
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

def evaluate_models(models, X_test, y_test, feature_names, args):
    """
    Evaluate trained models and visualize results.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    feature_names : list
        Names of features
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    logging.info("Starting model evaluation phase...")
    
    try:
        # Compare models
        logging.info("Comparing models...")
        comparison_results = compare_models(models, X_test, y_test)
        print("\nModel Comparison Results:")
        print(comparison_results)
        
        # Create comparison visualizations
        if not args.skip_visualization:
            logging.info("Creating model comparison visualizations...")
            fig, results_df = compare_models_plot(models, X_test, y_test)
            
            # Save comparison plot if requested
            if args.save_plots:
                output_dir = args.plot_dir if args.plot_dir else 'evaluation_results'
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        
        # Generate detailed evaluation for each model
        evaluation_results = {}
        for name, model in models.items():
            logging.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_regression_metrics(y_test, y_pred)
            print(f"\nEvaluation Metrics for {name}:")
            print_metrics_report(metrics, model_name=name)
            
            # Generate visualizations if not skipped
            if not args.skip_visualization:
                # Create and display/save evaluation plots
                if args.save_plots:
                    output_dir = args.plot_dir if args.plot_dir else 'evaluation_results'
                    save_evaluation_results(
                        model, X_test, y_test, feature_names,
                        model_name=name, output_dir=output_dir
                    )
                else:
                    # Just show key plots
                    plot_actual_vs_predicted(y_test, y_pred, model_name=name)
                    plot_residuals(y_test, y_pred, model_name=name)
                    
                    # Show feature importance if applicable
                    try:
                        plot_feature_importance(model, feature_names)
                    except Exception as e:
                        logging.info(f"Feature importance visualization not available for {name}: {str(e)}")
            
            evaluation_results[name] = metrics
        
        logging.info("Model evaluation complete")
        return {
            'comparison': comparison_results,
            'individual_results': evaluation_results
        }
    
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Social Media Engagement Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Workflow control arguments
    parser.add_argument('--run-all', action='store_true',
                        help='Run the complete pipeline from data generation to evaluation')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic social media data')
    parser.add_argument('--explore-data', action='store_true',
                        help='Run exploratory data analysis')
    parser.add_argument('--train-models', action='store_true',
                        help='Train machine learning models')
    parser.add_argument('--evaluate-models', action='store_true',
                        help='Evaluate trained models')
    
    # Data generation parameters
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--missing-pct', type=float, default=0.05,
                        help='Percentage of missing values to introduce')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise level for data generation')
    
    # Input/output parameters
    parser.add_argument('--input', type=str, default='data/social_media_data.csv',
                        help='Input data file path (CSV)')
    parser.add_argument('--output', type=str, default='data/social_media_data.csv',
                        help='Output file path for generated data (CSV)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save generated data to CSV file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load models')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models to disk')
    parser.add_argument('--load-models', action='store_true',
                        help='Load models from disk instead of training')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots instead of displaying them')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory to save plots')
    
    # Preprocessing parameters
    parser.add_argument('--skip-imputation', action='store_true',
                        help='Skip missing value imputation')
    parser.add_argument('--skip-outlier-treatment', action='store_true',
                        help='Skip outlier detection and treatment')
    parser.add_argument('--skip-scaling', action='store_true',
                        help='Skip feature scaling')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Skip feature engineering')
    parser.add_argument('--imputation-strategy', type=str, default='mean',
                        choices=['mean', 'median', 'mode', 'knn'],
                        help='Strategy for handling missing values')
    parser.add_argument('--outlier-method', type=str, default='iqr',
                        choices=['iqr', 'isolation_forest'],
                        help='Method for outlier detection')
    parser.add_argument('--outlier-treatment', type=str, default='clip',
                        choices=['clip', 'remove', 'flag'],
                        help='Treatment method for outliers')
    parser.add_argument('--scaling-method', type=str, default='standard',
                        choices=['standard', 'minmax', 'robust'],
                        help='Method for feature scaling')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Proportion of data for validation')
    
    # Model training parameters
    parser.add_argument('--models', nargs='+', 
                        default=['linear', 'random_forest', 'gradient_boosting'],
                        choices=['linear', 'random_forest', 'gradient_boosting'],
                        help='Models to train')
    parser.add_argument('--rf-trees', type=int, default=100,
                        help='Number of trees for Random Forest')
    parser.add_argument('--rf-depth', type=int, default=None,
                        help='Maximum depth for Random Forest trees')
    parser.add_argument('--gb-trees', type=int, default=100,
                        help='Number of trees for Gradient Boosting')
    parser.add_argument('--gb-lr', type=float, default=0.1,
                        help='Learning rate for Gradient Boosting')
    parser.add_argument('--gb-depth', type=int, default=3,
                        help='Maximum depth for Gradient Boosting trees')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--tune-model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'linear'],
                        help='Model to tune hyperparameters for')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search for hyperparameter tuning (default: random search)')
    parser.add_argument('--random-iter', type=int, default=10,
                        help='Number of iterations for random search')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    # Visualization parameters
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization generation')
    
    # Logging parameters
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the social media engagement prediction workflow.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log_file
    setup_logging(log_level, log_file)
    
    # Print welcome message
    logging.info("=" * 80)
    logging.info("Social Media Engagement Prediction Pipeline")
    logging.info("=" * 80)
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Track timing
        start_time = time.time()
        df = None
        models = {}
        
        # Determine what to run
        run_generate = args.generate_data or args.run_all
        run_explore = args.explore_data or args.run_all
        run_train = args.train_models or args.run_all
        run_evaluate = args.evaluate_models or args.run_all
        
        # Step 1: Generate or load data
        if run_generate:
            df = generate_data(args)
        else:
            # Load data if not generating
            if os.path.exists(args.input):
                logging.info(f"Loading data from {args.input}")
                df = pd.read_csv(args.input)
                logging.info(f"Loaded data with shape: {df.shape}")
            else:
                logging.error(f"Input file not found: {args.input}")
                if not run_generate:
                    logging.error("Use --generate-data to create a new dataset")
                    return 1
        
        # Make sure we have data for subsequent steps
        if df is None:
            logging.error("No data available. Exiting.")
            return 1
        
        # Step 2: Exploratory data analysis
        if run_explore:
            eda_results = explore_data(df, args)
            # Log key statistics from EDA results
            logging.info(f"EDA completed: {len(eda_results['stats'])} statistical measures calculated")
        
        # Step 3: Preprocess data
        if run_train or run_evaluate:
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, args)
            feature_names = X_train.columns.tolist()
        
        # Step 4: Train models
        if run_train:
            if args.load_models:
                # Load models from disk
                model_dir = args.model_dir if args.model_dir else 'models'
                if os.path.exists(model_dir):
                    logging.info(f"Loading models from {model_dir}")
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                    for model_file in model_files:
                        model_path = os.path.join(model_dir, model_file)
                        model = load_model(model_path)
                        model_name = os.path.splitext(model_file)[0].replace('_', ' ').title()
                        models[model_name] = model
                else:
                    logging.warning(f"Model directory not found: {model_dir}")
                    logging.info("Training new models instead")
                    models = train_models(X_train, y_train, X_val, y_val, args)
            else:
                # Train new models
                models = train_models(X_train, y_train, X_val, y_val, args)
        
        # Step 5: Evaluate models
        if run_evaluate and models:
            evaluation_results = evaluate_models(models, X_test, y_test, feature_names, args)
            # Log best model based on evaluation results
            best_model = min(evaluation_results['comparison'].items(), key=lambda x: x[1]['MSE'])[0]
            logging.info(f"Best performing model: {best_model} with MSE: {evaluation_results['comparison'][best_model]['MSE']:.4f}")
        
        # Calculate total runtime
        total_time = time.time() - start_time
        logging.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error in pipeline execution: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Example usage:
    # python main.py --run-all --save-data --save-plots
    # python main.py --generate-data --samples 2000 --save-data
    # python main.py --explore-data --input data/social_media_data.csv
    # python main.py --train-models --tune-hyperparams --cross-validate
    # python main.py --evaluate-models --save-plots
    sys.exit(main())