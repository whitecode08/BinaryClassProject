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
import optuna  

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce Optuna verbosity to avoid cluttering logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    """
    Handles Model Training AND Hyperparameter Tuning.
    """

    def __init__(self, config_path, data_config_path):
        self.config = self._load_config(config_path)
        self.data_config = self._load_config(data_config_path)
        
        self.model_conf = self.config['model']
        self.tuning_conf = self.config['tuning']  # Load tuning config
        self.eval_conf = self.config.get('evaluation', {})
        self.dirs = self.config['directories']
        
        # Ensure directories exist
        os.makedirs(self.dirs['model_save_dir'], exist_ok=True)
        os.makedirs(self.dirs['metrics_dir'], exist_ok=True)
        os.makedirs(self.dirs['plots_dir'], exist_ok=True)

        # Identify columns
        self.target_col = self.data_config['preprocess']['column_types']['target']
        self.id_col = self.data_config['preprocess']['column_types'].get('id_column')
        self.date_cols = self.data_config['preprocess']['column_types'].get('date_features', [])

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """Loads and prepares data."""
        data_dir = self.data_config['directories']['processed_data_dir']
        
        logger.info(f"Loading data from {data_dir}...")
        try:
            train_df = pd.read_parquet(os.path.join(data_dir, "train/train.parquet"))
            val_df = pd.read_parquet(os.path.join(data_dir, "val/val.parquet"))
            test_df = pd.read_parquet(os.path.join(data_dir, "test/test.parquet"))
            
            cols_to_drop = [self.target_col]
            if self.id_col: cols_to_drop.append(self.id_col)
            if self.date_cols: cols_to_drop.extend(self.date_cols)

            X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
            y_train = train_df[self.target_col]
            
            X_val = val_df.drop(columns=cols_to_drop, errors='ignore')
            y_val = val_df[self.target_col]
            
            X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
            y_test = test_df[self.target_col]
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except FileNotFoundError as e:
            logger.error(f"Missing parquet file: {e}")
            raise

    # ==========================================
    #  Applying Bayesian Hyperparameter Tuning
    # ==========================================
    def _get_search_space(self, trial):
        """
        Parses the YAML search_space and maps it to Optuna trial suggestions.
        """
        params = {}
        space = self.tuning_conf['search_space']

        for name, config in space.items():
            param_type = config['type']
            
            if param_type == 'int':
                params[name] = trial.suggest_int(name, config['low'], config['high'])
            
            elif param_type == 'uniform':
                params[name] = trial.suggest_float(name, config['low'], config['high'])
            
            elif param_type == 'loguniform':
                params[name] = trial.suggest_float(name, config['low'], config['high'], log=True)
            
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(name, config['choices'])

        return params

    def _objective(self, trial, X_train, y_train):
        """
        The Objective Function for Optuna.
        Uses Cross-Validation (cv_folds) to evaluate parameters.
        """
        # 1. Suggest Parameters
        params = self._get_search_space(trial)
        
        # 2. Add Fixed System Params (device, tree_method, etc.)
        params.update(self.model_conf['system'])
        params['objective'] = self.model_conf['objective']
        params['eval_metric'] = self.tuning_conf['metric_to_optimize']
        
        # 3. Create DMatrix for CV
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # 4. Run Cross-Validation
        cv_folds = self.tuning_conf.get('cv_folds', 3)
        early_stop = self.tuning_conf.get('early_stop_rounds', 10)
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=cv_folds,
            stratified=True,
            metrics=self.tuning_conf['metric_to_optimize'],
            early_stopping_rounds=early_stop,
            seed=42,
            verbose_eval=False 
        )

        # 5. Return the best score
        # xgb.cv returns a dataframe. We take the last (best) value.
        metric_name = f"test-{params['eval_metric']}-mean"
        best_score = cv_results[metric_name].iloc[-1]
        
        return best_score

    def run_tuning(self, X_train, y_train):
        """
        Runs the Bayesian Optimization loop using Optuna.
        """
        logger.info(">>> Starting Bayesian Hyperparameter Tuning...")
        
        # 1. Setup Storage (SQLite) to make it Resumable
        db_name = "optuna.db"
        storage_name = f"sqlite:///{db_name}"
        study_name = f"{self.model_conf['name']}_optimization"

        # 2. Create/Load Study
        direction = self.tuning_conf.get('direction', 'maximize')
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            direction=direction, 
            load_if_exists=True
        )

        logger.info(f"Study '{study_name}' loaded. Existing trials: {len(study.trials)}")
        
        # 3. Calculate remaining trials
        max_trials = self.tuning_conf.get('max_trials', 20)
        remaining_trials = max_trials - len(study.trials)

        if remaining_trials > 0:
            logger.info(f"Running {remaining_trials} more trials...")
            
            # Use partial to pass data to objective
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train), 
                n_trials=remaining_trials,
                show_progress_bar=True
            )
        else:
            logger.info("Max trials reached. Skipping optimization.")

        # 4. Get Best Params
        best_params = study.best_params
        logger.info(f"Best Params Found: {best_params}")
        logger.info(f"Best Score: {study.best_value}")
        
        return best_params

    # ==========================================
    #  Training and Evaluation Logic
    # ==========================================

    def evaluate_model(self, model, X_test, y_test):
        logger.info("Evaluating model on Test Set...")
        threshold = self.eval_conf.get('threshold', 0.5)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
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

        if "confusion_matrix" in self.eval_conf.get('report_metrics', []):
            self._plot_confusion_matrix(y_test, y_pred)

        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            plt.figure(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, values_format='d')
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(self.dirs['plots_dir'], "confusion_matrix.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

    def generate_plots(self, model, X_val):
        # Feature Importance
        if self.config['reporting']['feature_importance']['enabled']:
            imp_type = self.config['reporting']['feature_importance']['importance_type']
            plt.figure(figsize=(10, 8))
            xgb.plot_importance(model, importance_type=imp_type, max_num_features=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots_dir'], "feature_importance.png"))
            plt.close()

        # SHAP
        if self.config['reporting']['shap']['enabled']:
            # logger.info("Calculating SHAP values...") # Verbose
            sample_size = self.config['reporting']['shap']['background_sample_size']
            if len(X_val) > sample_size:
                X_sample = X_val.sample(sample_size, random_state=42)
            else:
                X_sample = X_val
            
            # Check if model is fitted before using TreeExplainer
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
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.model_conf['name'])
        mlflow.xgboost.autolog(log_models=True)
        
        with mlflow.start_run():
            # 1. Load Data
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
            
            # 2. Check for Tuning
            final_params = self.model_conf['params'].copy() # Start with defaults
            
            if self.tuning_conf.get('enabled', False):
                # Run Optimization
                tuned_params = self.run_tuning(X_train, y_train)
                # Merge Tuned Params into Final Params
                final_params.update(tuned_params)
            else:
                logger.info("Tuning disabled. Using default parameters from config.")

            # 3. Add System & Logic Params (that aren't tuned)
            final_params.update(self.model_conf['system'])
            final_params['objective'] = self.model_conf['objective']
            final_params['eval_metric'] = self.model_conf['eval_metrics']
            
            # Auto Scale Pos Weight if needed
            if final_params.get('scale_pos_weight') == 'auto':
                ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
                final_params['scale_pos_weight'] = ratio

            # Early Stopping Setup
            early_stop = self.tuning_conf.get('early_stop_rounds', 10)
            final_params['early_stopping_rounds'] = early_stop
            
            # 4. Train Final Model
            logger.info(">>> Training Final Model with Best Parameters...")
            logger.info(f"Final Params: {final_params}")
            
            model = xgb.XGBClassifier(**final_params)
            
            # We set verbose to 100 to reduce log noise, as requested
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=100  
            )

            # 5. Evaluate & Save
            self.evaluate_model(model, X_test, y_test)
            self.generate_plots(model, X_val)
            self.save_model(model)
            
            mlflow.log_artifacts(self.dirs['plots_dir'], artifact_path="plots")
            mlflow.log_artifacts(self.dirs['metrics_dir'], artifact_path="metrics")
            
            logger.info(">>> Pipeline Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    args = parser.parse_args()

    trainer = ModelTrainer(config_path=args.config, data_config_path=args.data_config)
    trainer.run()