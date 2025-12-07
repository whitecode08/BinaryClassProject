import argparse
import yaml
import pandas as pd
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. Logging Setup ---
# Configure logging to display time, severity level, and message.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles the data ingestion process: loading raw data, splitting it into 
    train/validation/test sets, and saving them as parquet files.
    
    Attributes:
        config (dict): Dictionary containing the full configuration.
        split_config (dict): Sub-dictionary for splitting parameters.
        dir_config (dict): Sub-dictionary for directory paths.
    """

    def __init__(self, config_path: str):
        """
        Initialize the DataIngestion class by loading the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.split_config = self.config['splitting']
        self.dir_config = self.config['directories']
        
        # Ensure output directory exists immediately
        os.makedirs(self.dir_config['processed_data_dir'], exist_ok=True)

    def _load_config(self, path: str) -> dict:
        """
        Internal method to safely load a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML content.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
            
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.error(f"Error parsing YAML file: {exc}")
                raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads the preprocessed dataset (parquet) from the featured directory 
        as defined in data_preprocessing.py output.

        Returns:
            pd.DataFrame: The loaded dataset.
        
        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        # Updated source: now reading from 'featured_data_dir'
        featured_path = self.dir_config['featured_data_dir']
        
        # The filename matches the output from data_preprocessing.py
        file_path = os.path.join(featured_path, "featured_data.parquet")
        
        logger.info(f"Attempting to load data from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Featured data file missing: {file_path}")
            logger.error("Please run 'src/data_preprocessing.py' first to generate this file.")
            raise FileNotFoundError(f"File {file_path} not found.")
        
        # Load Parquet
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet file: {e}")
            raise

    def split_data(self, df: pd.DataFrame):
        """
        Splits the dataframe into Training, Validation, and Test sets.
        Supports both Time-Series (chronological) and Stratified/Random splitting.

        Logic:
            1. Train set size = total * train_split_ratio
            2. Remaining data (Test + Val) = total - Train set
            3. Validation set size = Remaining * validation_ratio
            4. Test set size = Remaining - Validation set

        Args:
            df (pd.DataFrame): The raw dataframe to split.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        train_ratio = self.split_config['train_split_ratio']
        val_ratio = self.split_config['validation_ratio']
        
        target_col = self.split_config.get('stratify_col')
        time_col = self.split_config.get('time_col')

        logger.info("Starting data splitting process...")
        logger.info(f"Configuration - Train Ratio: {train_ratio}, Val Ratio (of remainder): {val_ratio}")

        # --- STRATEGY 1: Time Series Split ---
        # Used if a time column is defined and exists in the dataframe
        if time_col and time_col in df.columns:
            logger.info(f"Applying Time-Series Split based on column: '{time_col}'")
            
            # Sort by time to ensure past data is used to predict future
            df = df.sort_values(by=time_col).reset_index(drop=True)
            
            n_total = len(df)
            train_end = int(n_total * train_ratio)
            
            # First Split: Train vs (Val + Test)
            train_df = df.iloc[:train_end].copy()
            remainder_df = df.iloc[train_end:].copy()
            
            # Second Split: Val vs Test
            val_end = int(len(remainder_df) * val_ratio)
            val_df = remainder_df.iloc[:val_end].copy()
            test_df = remainder_df.iloc[val_end:].copy()

        # --- STRATEGY 2: Stratified / Random Split ---
        else:
            if target_col and target_col in df.columns:
                logger.info(f"Applying Stratified Split based on target: '{target_col}'")
                stratify_labels = df[target_col]
            else:
                logger.info("Applying Random Split (No stratify column found or defined)")
                stratify_labels = None

            # Split 1: Separate Train from (Val + Test)
            train_df, remainder_df = train_test_split(
                df, 
                train_size=train_ratio, 
                stratify=stratify_labels, 
                random_state=42,
                shuffle=True
            )

            # Prepare stratify labels for the second split if necessary
            if stratify_labels is not None:
                stratify_remainder = remainder_df[target_col]
            else:
                stratify_remainder = None

            # Split 2: Separate Validation from Test
            val_df, test_df = train_test_split(
                remainder_df, 
                train_size=val_ratio, 
                stratify=stratify_remainder, 
                random_state=42,
                shuffle=True
            )

        logger.info("Splitting complete.")
        logger.info(f"Train set shape: {train_df.shape}")
        logger.info(f"Val set shape:   {val_df.shape}")
        logger.info(f"Test set shape:  {test_df.shape}")
        
        return train_df, val_df, test_df

    def save_artifacts(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Saves the split dataframes into Parquet format for efficiency.
        
        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Testing data.
        """
        output_dir = self.dir_config['processed_data_dir']
        train_dir = self.dir_config['train_path']
        val_dir = self.dir_config['val_path']
        test_dir = self.dir_config['test_path']
        
        # Define paths
        train_path = os.path.join(output_dir, train_dir)
        val_path = os.path.join(output_dir, val_dir)
        test_path = os.path.join(output_dir, test_dir)

        logger.info(f"Saving split datasets to: {output_dir}")
        
        try:
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)
            logger.info("Artifacts saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            raise

    def run(self):
        """
        Main execution method for the pipeline stage.
        """
        try:
            logger.info(">>> Stage: Data Ingestion Started")
            
            # 1. Load
            df = self.load_data()
            
            # 2. Split
            train_df, val_df, test_df = self.split_data(df)
            
            # 3. Save
            self.save_artifacts(train_df, val_df, test_df)
            
            logger.info(">>> Stage: Data Ingestion Completed Successfully")
            
        except Exception as e:
            logger.error(">>> Stage: Data Ingestion Failed")
            logger.error(e)
            raise e

if __name__ == "__main__":
    # Argument Parser to handle config path input
    parser = argparse.ArgumentParser(description="Data Ingestion Pipeline Stage")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Path to config file")
    
    args = parser.parse_args()

    # Initialize and run
    ingestion = DataIngestion(config_path=args.config)
    ingestion.run()