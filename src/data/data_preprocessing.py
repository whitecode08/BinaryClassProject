# 1. Import Libraries
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
import os
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 2. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Create End to End Data Preprocessing Class
class DataPreprocessor:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.preprocess_config = self.config['preprocess']
        self.dir_config = self.config['directories']
        
        # Define Columns
        self.target_col = self.preprocess_config['column_types']['target']
        self.num_cols = self.preprocess_config['column_types']['numerical_features']
        self.cols_to_drop = self.preprocess_config['column_types']['features_to_drop']
        
        # Storage for fitted transformers (for future inference purposes)
        self.transformers = {}

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self, filepath):
        logger.info(f"Loading data from {filepath}")
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format")

    def drop_unwanted_features(self, df):
        logger.info(f"Dropping columns: {self.cols_to_drop}")
        # Only drop if columns exist in the dataframe
        existing_cols = [c for c in self.cols_to_drop if c in df.columns]
        return df.drop(columns=existing_cols)

    def handle_missing_values(self, df):
        """
        Handles missing values based on 'active_method' configuration.
        """
        imputation_conf = self.preprocess_config['imputation']['numerical']
        method = imputation_conf['active_method']
        logger.info(f"Handling missing values using method: {method}")

        # Filter numerical columns existing in the DF
        valid_num_cols = [c for c in self.num_cols if c in df.columns]

        if method in ['mean', 'median']:
            strategy = imputation_conf['options'][method]['strategy']
            imputer = SimpleImputer(strategy=strategy)
        
        elif method == 'knn':
            params = imputation_conf['options']['knn']
            imputer = KNNImputer(n_neighbors=params['n_neighbors'], weights=params['weights'])
        
        else:
            logger.warning("No valid imputation method specified. Skipping.")
            return df

        # Fit & Transform
        df[valid_num_cols] = imputer.fit_transform(df[valid_num_cols])
        
        # Save imputer for future use
        self.transformers['imputer'] = imputer
        return df

    def handle_outliers(self, df):
        """
        Handles outliers based on configuration.
        Note: Outlier handling is usually done ONLY on numerical features.
        """
        outlier_conf = self.preprocess_config['outliers']
        method = outlier_conf['active_method']
        logger.info(f"Handling outliers using method: {method}")

        valid_num_cols = [c for c in self.num_cols if c in df.columns]

        if method == 'capping':
            # Winsorization / Capping
            threshold = outlier_conf['options']['capping']['threshold']
            lower_quantile = threshold
            upper_quantile = 1.0 - threshold
            
            for col in valid_num_cols:
                lower_bound = df[col].quantile(lower_quantile)
                upper_bound = df[col].quantile(upper_quantile)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        elif method == 'iqr':
            multiplier = outlier_conf['options']['iqr']['multiplier']
            for col in valid_num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (multiplier * IQR)
                upper_bound = Q3 + (multiplier * IQR)
                # Option: Capping (Clip) so data is not lost
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        elif method == 'z_score':
            threshold = outlier_conf['options']['z_score']['threshold']
            for col in valid_num_cols:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                df[col] = np.clip(df[col], lower_bound, upper_bound)

        return df

    def scale_features(self, df):
        """
        Data Normalization/Standardization.
        """
        scaling_conf = self.preprocess_config['scaling']
        method = scaling_conf['active_method']
        logger.info(f"Scaling features using method: {method}")

        valid_num_cols = [c for c in self.num_cols if c in df.columns]

        if method == 'minmax':
            feat_range = tuple(scaling_conf['options']['minmax']['feature_range'])
            scaler = MinMaxScaler(feature_range=feat_range)
        elif method == 'standard':
            scaler = StandardScaler(
                with_mean=scaling_conf['options']['standard']['with_mean'],
                with_std=scaling_conf['options']['standard']['with_std']
            )
        elif method == 'robust':
            q_range = tuple(scaling_conf['options']['robust']['quantile_range'])
            scaler = RobustScaler(quantile_range=q_range)
        else:
            return df

        df[valid_num_cols] = scaler.fit_transform(df[valid_num_cols])
        self.transformers['scaler'] = scaler
        return df

    def save_data(self, df, output_dir, filename="featured_data.parquet"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        logger.info(f"Saving processed data to {path}")
        df.to_parquet(path, index=False)
        
        # Save transformers (scaler/imputer)
        transformers_path = os.path.join(output_dir, "transformers.joblib")
        joblib.dump(self.transformers, transformers_path)
        logger.info(f"Transformers saved to {transformers_path}")

    def run(self):
        # 1. Determine input path (can be from raw or argument)
        # As requested: from directories config -> raw_data_path
        # Assumption: filename 'dummy_dataset.csv' is hardcoded or obtained from dir list
        raw_dir = self.dir_config['raw_data_path']
        input_file = os.path.join(raw_dir, "dummy_dataset.csv") # Adjust if filename differs
        
        # 2. Execution Flow
        df = self.load_data(input_file)
        df = self.drop_unwanted_features(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.scale_features(df)
        
        # 3. Save
        self.save_data(df, self.dir_config['featured_data_dir'])

# 4. Code Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml", help="Path to config file")
    args = parser.parse_args()

    processor = DataPreprocessor(config_path=args.config)
    processor.run()