import data
import features
import models
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import os
import time


class LastNMonthsTimeSeriesSplit:
    def __init__(self, n_validation_months=6):
        """
        n_validation_months: Number of months at the end of the series to use as sequential test sets
        """
        self.n_validation_months = n_validation_months

    def split(self, X, y=None, groups=None):
        """
        Using the 'groups' argument to safely pass dates ensures parallel 
        jobs (n_jobs > 1) map indices correctly.
        """
        if groups is None:
            raise ValueError("You must pass the date column/series to the 'groups' parameter of grid_search.fit()")
            
        # Convert groups to datetime safely
        dates = pd.to_datetime(groups)
        
        # 1. Identify unique Year-Months present in the dataset and sort them
        year_months = sorted(dates.dt.to_period('M').unique())
        
        # 2. Extract the newest N months earmarked for evaluation
        test_months = year_months[-self.n_validation_months:]
        
        # Convert date series to Period for alignment matching
        data_periods = dates.dt.to_period('M')
        
        # Use underlying integer positions independent of the dataframe's index
        indices = np.arange(len(X))
        
        # 3. Generate splits
        for current_test_month in test_months:
            test_mask = (data_periods == current_test_month).values
            train_mask = (data_periods < current_test_month).values
            
            test_idx = indices[test_mask]
            train_idx = indices[train_mask]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_validation_months


def tune():
    start_time = time.time()
    listing = data.load("listings")
    listing_sorted = listing.sort_values('date_listed').reset_index(drop=True)

    X = listing_sorted.drop(columns='sold_price')
    X_om = listing_sorted.drop(columns=['sold_price', 'list_price'])
    Y = listing_sorted['sold_price']

    # Extract the alignment series to pass to 'groups'
    dates_vector = listing_sorted['date_listed']

    custom_cv = LastNMonthsTimeSeriesSplit(n_validation_months=6)

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [100, 500, 1000],
        'max_leaf_nodes': [15, 31, 63],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0.0, 0.1, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=models.XGBModel(features.feature_list_mlp_off_market, features.get_preprocessor_random_forest('off_market'), random_state=42),
        param_grid=param_grid,
        cv=custom_cv,               # <--- Inject custom validation strategy here
        scoring='neg_mean_squared_error',
        n_jobs=5,
        verbose=1
    )

    grid_search.fit(X_om, Y, groups=dates_vector)

    results_df = pd.DataFrame(grid_search.cv_results_)

    # Convert the negative MSE score into positive metrics
    results_df['mean_mse'] = -results_df['mean_test_score']
    results_df['mean_rmse'] = np.sqrt(results_df['mean_mse'])  # RMSE is in the original target units ($)

    results_df_sorted = results_df.sort_values(by='rank_test_score')

    
    # 4. Generate a unique, timestamped filename to prevent overwriting past runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"grid_search_results_{timestamp}.csv"

    # 5. Export to CSV
    results_df_sorted.to_csv(output_filename, index=False)

    print(f"✅ Grid search results successfully saved to: {os.path.abspath(output_filename)}")

    # 2. Capture the end time and calculate the duration
    end_time = time.time()
    total_duration_seconds = end_time - start_time
    
    # 3. Format into a clean, human-readable layout
    minutes, seconds = divmod(total_duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n==========================================")
    print(f"⏱️  TOTAL RUNTIME: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("==========================================")

tune()
