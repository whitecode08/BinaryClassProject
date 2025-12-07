import argparse
import yaml
import pandas as pd
import numpy as np
import logging
import os
import json
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the model training process without hyperparameter tuning.
    Trains a single XGBoost model based on config.yaml.
    """

    def __init__(self, config_path, data_config_path):
        self.config = self._load_config(config_path)
        self.data_config = self._load_config(data_config_path)
        
        self.model_conf = self.config['model']
        self.dirs = self.config['directories']
        
        # Ensure directories exist
        os.makedirs(self.dirs['model_save_dir'], exist_ok=True)
        os.makedirs(self.dirs['metrics_dir'], exist_ok=True)
        os.makedirs(self.dirs['plots_dir'], exist_ok=True)

        # Identify columns to exclude from training (Target, ID, Dates)
        self.target_col = self.data_config['preprocess']['column_types']['target']
        self.id_col = self.data_config['preprocess']['column_types'].get('id_column')
        self.date_cols = self.data_config['preprocess']['column_types'].get('date_features', [])

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """
        Loads processed data from parquet files and removes non-feature columns.
        """
        data_dir = self.data_config['directories']['processed_data_dir']
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} does not exist. Please run 'src/data/data_ingestion.py' first.")

        logger.info(f"Loading data from {data_dir}...")
        try:
            train_df = pd.read_parquet(os.path.join(data_dir, "train/train.parquet"))
            val_df = pd.read_parquet(os.path.join(data_dir, "val/val.parquet"))
            test_df = pd.read_parquet(os.path.join(data_dir, "test/test.parquet"))
            
            # Define columns to drop (Target + ID + Dates)
            cols_to_drop = [self.target_col]
            if self.id_col:
                cols_to_drop.append(self.id_col)
            if self.date_cols:
                cols_to_drop.extend(self.date_cols)

            logger.info(f"Dropping non-feature columns for training: {cols_to_drop}")

            # Separate Features (X) and Target (y)
            # errors='ignore' ensures no error if a column is already missing
            X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
            y_train = train_df[self.target_col]
            
            X_val = val_df.drop(columns=cols_to_drop, errors='ignore')
            y_val = val_df[self.target_col]
            
            X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
            y_test = test_df[self.target_col]
            
            logger.info(f"Data Loaded. Train shape: {X_train.shape}")
            # Verify no object columns remain
            obj_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            if obj_cols:
                logger.warning(f"Warning: Object columns still present in X_train: {obj_cols}. This might cause XGBoost errors.")

            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except FileNotFoundError as e:
            logger.error(f"Missing parquet file: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        logger.info("Evaluating model on Test Set...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "log_loss": float(log_loss(y_test, y_prob))
        }
        
        metrics_path = os.path.join(self.dirs['metrics_dir'], "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Test Metrics: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
        return metrics

    def generate_plots(self, model, X_val):
        # Feature Importance
        if self.config['reporting']['feature_importance']['enabled']:
            imp_type = self.config['reporting']['feature_importance']['importance_type']
            plt.figure(figsize=(10, 8))
            xgb.plot_importance(model, importance_type=imp_type, max_num_features=20)
            plt.title(f"Feature Importance ({imp_type})")
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots_dir'], "feature_importance.png"))
            plt.close()

        # SHAP
        if self.config['reporting']['shap']['enabled']:
            logger.info("Calculating SHAP values...")
            sample_size = self.config['reporting']['shap']['background_sample_size']
            
            if len(X_val) > sample_size:
                X_sample = X_val.sample(sample_size, random_state=42)
            else:
                X_sample = X_val
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots_dir'], "shap_summary.png"))
            plt.close()

    def save_model(self, model):
        filename = self.config['serialization']['filename_pattern'].format(version="v1")
        save_path = os.path.join(self.dirs['model_save_dir'], filename)
        logger.info(f"Saving model to {save_path}...")
        model.save_model(save_path)

    def run(self):
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.model_conf['name'])
        mlflow.xgboost.autolog()
        
        with mlflow.start_run():
            # 1. Load Data
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
            
            # 2. Prepare Params
            final_params = self.model_conf['params'].copy()
            final_params.update(self.model_conf['system'])
            final_params['objective'] = self.model_conf['objective']
            final_params['eval_metric'] = self.model_conf['eval_metrics']
            
            # Handle Auto Scale Pos Weight
            if final_params.get('scale_pos_weight') == 'auto':
                ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
                final_params['scale_pos_weight'] = ratio
                logger.info(f"Auto scale_pos_weight: {ratio:.2f}")

            # --- FIX: Move early_stopping_rounds to Constructor ---
            early_stop = self.config['tuning'].get('early_stop_rounds', 10)
            final_params['early_stopping_rounds'] = early_stop
            
            # 3. Train Model
            logger.info(f"Training XGBoost model (Early Stopping = {early_stop})...")
            
            # early_stopping_rounds is now passed here in **final_params
            model = xgb.XGBClassifier(**final_params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=100
            )

            # 4. Evaluate
            self.evaluate_model(model, X_test, y_test)
            
            # 5. Reports & Save
            self.generate_plots(model, X_val)
            self.save_model(model)
            
            # Log additional artifacts
            mlflow.log_artifacts(self.dirs['plots_dir'], artifact_path="plots")
            mlflow.log_artifacts(self.dirs['metrics_dir'], artifact_path="metrics")
            
            logger.info(">>> Model Training Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    args = parser.parse_args()

    trainer = ModelTrainer(config_path=args.config, data_config_path=args.data_config)
    trainer.run()