import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_social_media_data(n_samples=1000, output_file=None, random_seed=None, 
                              missing_data_pct=0.05, noise_level=0.1):
    """
    Generate synthetic social media engagement data.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic data samples to generate
    output_file : str
        Path to save the generated data (CSV format)
    random_seed : int or None
        Seed for random number generation to ensure reproducibility
    missing_data_pct : float
        Percentage of data to replace with NaN values (0.0 to 1.0)
    noise_level : float
        Amount of noise to add to the engagement_rate calculation (0.0 to 1.0)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the generated social media data
    """
    # Set seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate base data features
    data = {
        # Post length in characters (between 10 and 500)
        'post_length': np.random.randint(10, 500, size=n_samples),
        
        # Number of images in the post (0 to 10, with most posts having fewer images)
        'images_count': np.random.choice(range(11), size=n_samples, 
                                         p=[0.20, 0.35, 0.25, 0.10, 0.05, 0.02, 0.01, 0.01, 0.005, 0.004, 0.001]),
        
        # Number of hashtags (0 to 30, with most posts having moderate hashtag counts)
        'hashtags_count': np.random.choice(range(31), size=n_samples,
                                           p=[0.05] + [0.04] * 5 + [0.03] * 5 + [0.02] * 5 + 
                                             [0.01] * 10 + [0.005] * 5),
        
        # Followers count (log-normal distribution to simulate real-world follower counts)
        'followers_count': np.exp(np.random.normal(8, 2, size=n_samples)).astype(int),
    }
    
    # Generate post times distributed throughout the day
    # Create base datetime
    base_date = datetime(2023, 1, 1)
    # Generate random minutes to add (0 to 24*60 minutes in a day)
    minutes_to_add = np.random.randint(0, 24 * 60, size=n_samples)
    # Convert to hour of day (float) for easier modeling
    data['post_hour'] = [((base_date + timedelta(minutes=m)).hour + 
                         (base_date + timedelta(minutes=m)).minute / 60) 
                        for m in minutes_to_add]
    
    # Generate categorical features for day of week (0=Monday, 6=Sunday)
    data['post_day'] = np.random.randint(0, 7, size=n_samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate engagement rate with a realistic formula and add noise
    # Base formula components with different weights for each feature
    engagement = (
        # Longer posts tend to have less engagement after a certain length
        -0.00005 * (df['post_length'] - 100) ** 2 + 0.5 +
        
        # Images tend to increase engagement up to a point
        0.7 * np.log1p(df['images_count']) +
        
        # Hashtags increase visibility up to a point then reduce engagement
        0.4 * np.log1p(df['hashtags_count']) - 0.01 * df['hashtags_count']**2 +
        
        # Accounts with more followers tend to have lower engagement rates
        -0.3 * np.log10(df['followers_count'] + 1) + 2.0 +
        
        # Time of day effects (higher engagement in evenings and mornings)
        0.5 * np.sin((df['post_hour'] - 8) * np.pi / 12) +
        
        # Day of week effects (higher on weekends)
        0.3 * (df['post_day'] >= 5)
    )
    
    # Add random noise to make it more realistic
    noise = np.random.normal(0, noise_level, size=n_samples)
    
    # Calculate final engagement rate (percentage) and clip to realistic bounds
    df['engagement_rate'] = np.clip((engagement + noise) * 2, 0.1, 15.0)
    
    # Introduce missing values if specified
    if missing_data_pct > 0:
        # Calculate how many values to replace
        total_cells = df.size
        n_missing = int(total_cells * missing_data_pct)
        
        # Randomly select cells to replace with NaN
        # (excluding the target variable - engagement_rate)
        rows = np.random.randint(0, df.shape[0], size=n_missing)
        cols = np.random.choice(df.columns[df.columns != 'engagement_rate'], size=n_missing)
        
        # Replace selected cells with NaN
        for i in range(n_missing):
            df.loc[rows[i], cols[i]] = np.nan
    
    # Save to CSV if output_file is specified
    if output_file is not None:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage of the function
    df = generate_social_media_data(
        n_samples=1000,
        output_file="data/social_media_data.csv",
        random_seed=42,
        missing_data_pct=0.05,
        noise_level=0.1
    )
    
    # Display summary statistics
    print("Generated data summary:")
    print(df.describe())
    
    # Display missing value information
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
