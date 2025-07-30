#!/usr/bin/env python3
"""
LightGBM Model Training Script for Propaganda Detection
Cleaned and refactored version with configurable parameters
"""

import sqlite3
import time
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve
)
import lightgbm as lgb
from joblib import dump, load
import shap


class LightGBMTrainer:
    """
    A class to handle LightGBM model training with configurable parameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration parameters
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.db_path = config.get('db_path', 'ai_prop_dataset.db')
        self.table_name = config.get('table_name', 'ai_prop_liwc')
        # self.drop_features = config.get('drop_features', {})
        self.x_features = config.get('x_features', {})
        self.model_params = config.get('model_params', {})
        self.training_params = config.get('training_params', {})
        
    def execute_query_pandas(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as pandas DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame containing query results
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_classification_metrics(self, y_test: np.ndarray, y_predicted: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_test: True labels
            y_predicted: Predicted labels
            prob: Prediction probabilities
            
        Returns:
            DataFrame containing all metrics
        """
        # Basic metrics
        accuracy = round(accuracy_score(y_test, y_predicted), 2)
        precision = round(precision_score(y_test, y_predicted), 2)
        precision_0 = round(precision_score(y_test, y_predicted, pos_label=0), 2)
        recall = round(recall_score(y_test, y_predicted), 2)
        specificity = round(recall_score(y_test, y_predicted, pos_label=0), 2)
        f1 = round(f1_score(y_test, y_predicted), 2)
        f1_0 = round(f1_score(y_test, y_predicted, pos_label=0), 2)
        
        # AUC metrics
        p, r, th = precision_recall_curve(y_test, prob)
        pr_auc = round(auc(r, p), 2)
        
        if len(np.unique(y_test)) == 1:
            roc = np.NaN
        else:
            roc = round(roc_auc_score(y_test, prob), 2)
        
        # Rates
        pos_pred_rate = round(sum(y_predicted) * 100 / len(y_predicted), 2)
        pos_rate = round(sum(y_test) * 100 / len(y_test), 2)
        
        # Confusion matrix components
        temp = pd.DataFrame({'actual': y_test, 'prediction': y_predicted})
        tp = len(temp[(temp['actual'] == 1) & (temp['prediction'] == 1)])
        tn = len(temp[(temp['actual'] == 0) & (temp['prediction'] == 0)])
        fp = len(temp[(temp['actual'] == 0) & (temp['prediction'] == 1)])
        fn = len(temp[(temp['actual'] == 1) & (temp['prediction'] == 0)])
        
        metrics = {
            'metrics': [
                "Accuracy", "Precision", "Recall", "Specificity", "Precision_0",
                "F1", "F1_0", "PR_AUC", "ROC", "TP", "FP", "TN", "FN",
                "PredictionRate", "PositiveClassRate", "Count"
            ],
            'value': [
                accuracy, precision, recall, specificity, precision_0,
                f1, f1_0, pr_auc, roc, tp, fp, tn, fn,
                pos_pred_rate, pos_rate, len(y_test)
            ]
        }
        
        return pd.DataFrame(metrics)
    
    def prepare_features(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """
        Prepare features by removing specified columns and normalizing
        
        Args:
            df: Input DataFrame
            feature_set: Name of feature set to use
            
        Returns:
            Normalized feature DataFrame
        """
        if feature_set not in self.x_features:
            raise ValueError(f"Feature set '{feature_set}' not found in configuration")
        
        # remove_features = self.drop_features[feature_set]
        # X = df.drop(remove_features, axis=1)
        
        X = df[self.x_features[feature_set]].copy()

        for column in X.columns:
            max_val = X[column].max()
            min_val = X[column].min()
            if max_val == min_val:
                print(f"Warning: Column {column} has no variation and will be set to 0")
                X[column] = 0
            else:
                X[column] = (X[column] - min_val) / (max_val - min_val)

        return X
    
    def find_optimal_threshold(self, y_test: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Find optimal classification threshold using Youden's J statistic
        
        Args:
            y_test: True labels
            y_pred_prob: Prediction probabilities
            
        Returns:
            Optimal threshold value
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        threshold = thresholds[np.argmax(tpr - fpr)]
        return threshold
    
    def train_model(self, title: str, query_condition: str, target_column: str, 
                   feature_set: str = 'default') -> Dict[str, Any]:
        """
        Train LightGBM model with specified parameters
        
        Args:
            title: Description of the training scenario
            query_condition: SQL WHERE clause condition
            target_column: Name of target column
            feature_set: Feature set to use for training
            
        Returns:
            Dictionary containing model and results
        """
        print(f"\n{'='*60}")
        print(f"Training: {title}")
        print(f"{'='*60}")
        
        # Load data
        query = f'SELECT * FROM {self.table_name} {query_condition}'
        df = self.execute_query_pandas(query)
        
        # Prepare features and target
        X = self.prepare_features(df, feature_set)
        y = df[target_column]
        
        # Split data
        test_size = self.training_params.get('test_size', 0.3)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Train model
        model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=self.training_params.get('num_rounds', 100)
        )
        
        # Make predictions
        y_pred_prob = model.predict(X_test)
        
        # Find optimal threshold and make binary predictions
        threshold = self.find_optimal_threshold(y_test, y_pred_prob)
        y_pred_binary = (y_pred_prob >= threshold).astype(int)
        
        print(f'Optimal threshold: {threshold:.4f}')
        
        # Calculate metrics
        metrics_df = self.get_classification_metrics(y_test, y_pred_binary, y_pred_prob)
        print(metrics_df)
        
        # Save model if specified
        if self.config.get('save_model', False):
            model_filename = f"LGBM_{title.replace(' ', '_')}.pkl"
            dump(model, model_filename)
            print(f"Model saved as: {model_filename}")
        
        return {
            'model': model,
            'metrics': metrics_df,
            'threshold': threshold,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_prob': y_pred_prob,
            'y_pred_binary': y_pred_binary
        }
    
    def generate_shap_plots(self, model, X_data: pd.DataFrame, save_plots: bool = False):
        """
        Generate SHAP plots for model interpretation
        
        Args:
            model: Trained LightGBM model
            X_data: Feature data for SHAP analysis
            save_plots: Whether to save plots to files
        """
        if not self.config.get('generate_shap', False):
            return
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        
        # Summary plot
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]  # For binary classification
        else:
            shap_values_plot = shap_values
        
        shap.summary_plot(shap_values_plot, X_data, plot_type="dot", show=not save_plots)
        if save_plots:
            plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Feature importance plot
        shap.summary_plot(shap_values_plot, X_data, plot_type="bar", show=not save_plots)
        if save_plots:
            plt.savefig("shap_importance_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_feature_importance(self, model, max_features: int = 25):
        """
        Plot LightGBM feature importance
        
        Args:
            model: Trained LightGBM model
            max_features: Maximum number of features to display
        """
        if self.config.get('plot_importance', False):
            ax = lgb.plot_importance(model, max_num_features=max_features)
            plt.tight_layout()
            plt.show()


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for the trainer
    
    Returns:
        Default configuration dictionary
    """
    return {
        'db_path': 'ai_propaganda.db',
        'table_name': 'ai_prop_liwc',
        'x_features' : {
            'withPunc': [
                "Analytic", "Clout", "Authentic", "Tone", "WPS", "BigWords", "Dic", "Linguistic",
                "function", "pronoun", "ppron", "i", "we", "you", "shehe", "they", "ipron", "det",
                "article", "number", "prep", "auxverb", "adverb", "conj", "negate", "verb", "adj",
                "quantity", "Drives", "affiliation", "achieve", "power", "Cognition", "allnone",
                "cogproc", "insight", "cause", "discrep", "tentat", "certitude", "differ", "memory",
                "Affect", "tone_pos", "tone_neg", "emotion", "emo_pos", "emo_neg", "emo_anx",
                "emo_anger", "emo_sad", "swear", "Social", "socbehav", "prosocial", "polite", "conflict",
                "moral", "comm", "socrefs", "family", "friend", "female", "male", "Culture", "politic",
                "ethnicity", "tech", "Lifestyle", "leisure", "home", "work", "money", "relig",
                "Physical", "health", "illness", "wellness", "mental", "substances", "sexual", "food",
                "death", "need", "want", "acquire", "lack", "fulfill", "fatigue", "reward", "risk",
                "curiosity", "allure", "Perception", "attention", "motion", "space", "visual",
                "auditory", "feeling", "time", "focuspast", "focuspresent", "focusfuture",
                "Conversation", "netspeak", "assent", "nonflu", "filler", "AllPunc", "Period", "Comma",
                "QMark", "Exclam", "Apostro", "OtherP", "Emoji"
            ],
            'noPunc': [
                "Analytic", "Clout", "Authentic", "Tone", "WPS", "BigWords", "Dic", "Linguistic",
                "function", "pronoun", "ppron", "i", "we", "you", "shehe", "they", "ipron", "det",
                "article", "number", "prep", "auxverb", "adverb", "conj", "negate", "verb", "adj",
                "quantity", "Drives", "affiliation", "achieve", "power", "Cognition", "allnone",
                "cogproc", "insight", "cause", "discrep", "tentat", "certitude", "differ", "memory",
                "Affect", "tone_pos", "tone_neg", "emotion", "emo_pos", "emo_neg", "emo_anx",
                "emo_anger", "emo_sad", "swear", "Social", "socbehav", "prosocial", "polite", "conflict",
                "moral", "comm", "socrefs", "family", "friend", "female", "male", "Culture", "politic",
                "ethnicity", "tech", "Lifestyle", "leisure", "home", "work", "money", "relig",
                "Physical", "health", "illness", "wellness", "mental", "substances", "sexual", "food",
                "death", "need", "want", "acquire", "lack", "fulfill", "fatigue", "reward", "risk",
                "curiosity", "allure", "Perception", "attention", "motion", "space", "visual",
                "auditory", "feeling", "time", "focuspast", "focuspresent", "focusfuture",
                "Conversation", "netspeak", "assent", "nonflu", "filler", "Emoji"
            ]
        },
        # 'drop_features': { # Note: These are the variables which are REMOVED from the data set.
        #     'withPunc': [
        #         "Source", "Title", "Text", "Label", "Bias", "RightBias", "UserPrompt", "TargetWC", "AI", "articleID", "model", "Wave", "WC", "topic"
        #     ],
        #     'noPunc': [
        #         "Source", "Title", "Text", "Label", "Bias", "RightBias", "UserPrompt", "TargetWC", "AI", "articleID", "model", "Wave", "WC", "topic",
        #         "AllPunc", "Period", "Comma", "QMark", "Exclam", "Apostro", "OtherP"
        #     ]
            
        # },
        'model_params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'verbose': -1
        },
        'training_params': {
            'num_rounds': 100,
            'test_size': 0.3
        },
        'save_model': False,
        'plot_importance': False,
        'generate_shap': False
    }


def main():
    """
    Main function to run the training scenarios
    """
    # Create configuration
    config = create_default_config()
    
    # Initialize trainer
    trainer = LightGBMTrainer(config)
    
    # Define training scenarios with topics
    scenarios = [
        {
            'title': 'Combined_DetectHumanProp',
            'query_condition': 'WHERE WC >= 100 AND model LIKE "human"',
            'target_column': 'Label',
            'feature_set': 'withPunc'
        },
        # {
        #     'title': 'Combined_DetectAI_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model IN ("gpt-3.5-turbo", "human")',
        #     'target_column': 'AI',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT35',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT4o',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT4o_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT41',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'Combined_DetectAIProp_GPT41_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT35',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT4o',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT4o_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT41',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'climate_DetectAIProp_GPT41_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "climate"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT35',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT4o',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT4o_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT41',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'covid_DetectAIProp_GPT41_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "covid"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT35',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT4o',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT4o_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT41',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'capitolriot_DetectAIProp_GPT41_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "capitolriot"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT35',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT35_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-3.5-turbo" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT4o',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT4o_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4o" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT41',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'withPunc'
        # },
        # {
        #     'title': 'lgbt_DetectAIProp_GPT41_noPunc',
        #     'query_condition': 'WHERE WC >= 100 AND model LIKE "gpt-4.1" AND topic LIKE "lgbt"',
        #     'target_column': 'Label',
        #     'feature_set': 'noPunc'
        # },
    ]

    
    # Train models for each scenario
    results = {}
    for scenario in scenarios:
        result = trainer.train_model(
            title=scenario['title'],
            query_condition=scenario['query_condition'],
            target_column=scenario['target_column'],
            feature_set=scenario['feature_set']
        )
        results[scenario['title']] = result
        
        # Generate SHAP plots if enabled
        if config.get('generate_shap', False):
            trainer.generate_shap_plots(
                result['model'],
                result['X_test'],
                save_plots=True
            )
    
    return results


if __name__ == "__main__":
    results = main()