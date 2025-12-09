import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, config_path, data_config_path, model_path=None):
        self.config = self._load_yaml(config_path)
        self.data_config = self._load_yaml(data_config_path)
        
        # Paths
        self.data_dir = self.data_config['directories']['processed_data_dir']
        self.model_dir = self.config['directories']['model_save_dir']
        
        # If model_path is not provided explicitly, try to find the default one
        if model_path:
            self.model_path = model_path
        else:
            filename = self.config['serialization']['filename_pattern'].format(version="v1")
            self.model_path = os.path.join(self.model_dir, filename)

        # Columns to ignore during prediction (ID, Date, Target)
        self.target_col = self.data_config['preprocess']['column_types']['target']
        self.id_col = self.data_config['preprocess']['column_types'].get('id_column')
        self.date_cols = self.data_config['preprocess']['column_types'].get('date_features', [])
        
        # Load Model
        self.model = self._load_model()
        
        # Load Test Data (Lazy loading: only loaded when needed)
        self.test_df = None

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self):
        logger.info(f"Loading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Did you run training?")
        
        model = xgb.XGBClassifier()
        model.load_model(self.model_path)
        logger.info("Model loaded successfully.")
        return model

    def _load_test_data(self):
        if self.test_df is None:
            path = os.path.join(self.data_dir, "test/test.parquet")
            logger.info(f"Loading test data from: {path}")
            self.test_df = pd.read_parquet(path)
        return self.test_df

    def _prepare_single_row(self, row_series):
        """
        Cleans a single row of data to match the model's expected input.
        """
        # Convert series to dict, drop unused columns
        data = row_series.to_dict()
        
        cols_to_drop = [self.target_col]
        if self.id_col: cols_to_drop.append(self.id_col)
        if self.date_cols: cols_to_drop.extend(self.date_cols)
        
        # Create a DataFrame with 1 row
        df_input = pd.DataFrame([data])
        df_input = df_input.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure column order matches the model if possible (XGBoost is sensitive to this)
        # In a production app, you would enforce schema here.
        return df_input

    def predict_random_customer(self):
        """
        Picks a random customer from test.parquet and predicts churn.
        """
        df = self._load_test_data()
        
        # Pick random row
        random_idx = np.random.randint(0, len(df))
        row = df.iloc[random_idx]
        
        logger.info("\n" + "="*40)
        logger.info(f"Selected Customer Index: {random_idx}")
        if self.id_col in row:
            logger.info(f"Customer ID: {row[self.id_col]}")
        logger.info("="*40)

        # Show some key features (Top 5 just for display context)
        # You can adjust this to show specific columns you care about
        logger.info("--- Key Customer Features ---")
        display_cols = list(row.index)[:5] # Just show first 5 for brevity
        for col in display_cols:
            logger.info(f"{col}: {row[col]}")
        logger.info("... (and remaining features)")

        # Prepare input
        X_input = self._prepare_single_row(row)
        
        # Predict
        self._run_prediction(X_input, actual_label=row[self.target_col])

    def predict_manual_input(self):
        """
        Allows expert user to define a custom feature set in code or interactive (simplified).
        """
        logger.info("\n--- Manual Input Mode ---")
        logger.info("NOTE: This requires defining the dictionary in the code or a JSON file.")
        logger.info("For now, I will use a blank template based on your test data.")
        
        df = self._load_test_data()
        
        # Get one row to use as a template (structure only)
        template = self._prepare_single_row(df.iloc[0]).iloc[0].to_dict()
        
        # Here you would typically connect this to a UI or API input.
        # For this script, we will just simulate a slight modification of an average user.
        print("\nCreating a 'Synthetic Average Customer'...")
        
        # Create a dummy row with mean values (simple imputation for testing)
        # This is just to ensure the model doesn't crash on shape.
        X_input = df.drop(columns=[self.target_col], errors='ignore').mean().to_frame().T
        
        # You can manually override specific features here:
        # X_input['age'] = 50 
        # X_input['balance'] = 0
        
        self._run_prediction(X_input, actual_label=None)

    def _run_prediction(self, X_input, actual_label=None):
        # 1. Get Probability
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = self.model.predict_proba(X_input)[0]
        churn_prob = probs[1]
        
        # 2. Get Threshold (Default 0.5, usually loaded from config)
        threshold = self.config.get('evaluation', {}).get('threshold', 0.5)
        prediction = 1 if churn_prob >= threshold else 0
        
        # 3. Output
        logger.info("\n" + "-"*30)
        logger.info(f"PREDICTION RESULTS")
        logger.info("-"*30)
        logger.info(f"Churn Probability : {churn_prob:.4f}  ({churn_prob*100:.2f}%)")
        logger.info(f"Predicted Class   : {'CHURN (1)' if prediction == 1 else 'NO CHURN (0)'}")
        
        if actual_label is not None:
            result_match = "✅ Correct" if prediction == actual_label else "❌ Incorrect"
            logger.info(f"Actual Label      : {actual_label}")
            logger.info(f"Result            : {result_match}")
        logger.info("-"*30 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    args = parser.parse_args()

    try:
        tester = ModelTester(args.config, args.data_config)
        
        while True:
            print("\nSelect an action:")
            print("1. Test Random Customer (from Test Set)")
            print("2. Test Synthetic Average Customer")
            print("q. Quit")
            
            choice = input("Enter choice: ").strip().lower()
            
            if choice == '1':
                tester.predict_random_customer()
            elif choice == '2':
                tester.predict_manual_input()
            elif choice == 'q':
                break
            else:
                print("Invalid choice.")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()