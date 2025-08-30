import sqlite3
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import neural_network, svm, model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Configuration
MODEL = "gpt-3.5-turbo"
TOPIC = "climate" # Options: "lgbt", "climate", "covid", "capitolriot"
TARGET_VARIABLE = "Label"
X_FEATURES = "withPunc"  # Options: "withPunc", "noPunc"
PATH = 'ai_propaganda.db'
SEED = 42

# Model selection flags
RUN_DNN = True  
RUN_SNN = True  
RUN_SVM = True
RUN_LGBM = True

def execute_query_pandas(path: str, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as pandas DataFrame."""
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_feature_columns(x_features: str) -> list:
    """Return feature columns based on the selected feature set."""
    if x_features == "withPunc":
        return [
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
        ]
    elif x_features == "noPunc":
        return [
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
    else:
        raise ValueError("Invalid x_features value. Please ensure you have selected a valid model path.")

def prepare_features(df: pd.DataFrame, feature_columns: list, scaler: MinMaxScaler = None, fit_scaler: bool = True) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Prepare features by selecting specified columns and scaling
   
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names to use
        scaler: Pre-fitted MinMaxScaler (for test data)
        fit_scaler: Whether to fit the scaler (True for train, False for test)
       
    Returns:
        Tuple of (normalized feature DataFrame, fitted scaler)
    """
    X = df[feature_columns].copy()
    
    # Handle missing values
    X = X.fillna(0)
   
    # Initialize scaler if not provided
    if scaler is None:
        scaler = MinMaxScaler()
   
    if fit_scaler:
        # Fit and transform for training data
        X_scaled = scaler.fit_transform(X)
    else:
        # Only transform for test data (using scaler fitted on training data)
        X_scaled = scaler.transform(X)
   
    # Convert back to DataFrame with original column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
   
    return X_scaled_df, scaler

def load_and_preprocess_data() -> tuple:
    """Load data from database and perform preprocessing."""
    print("Loading and preprocessing data...")
    
    # Execute query
    query = f'SELECT * FROM ai_prop_liwc WHERE WC >= 100 AND model LIKE "{MODEL}" AND topic LIKE "{TOPIC}";'
    data = execute_query_pandas(PATH, query)
    
    # Get feature columns
    feature_columns = get_feature_columns(X_FEATURES)
    
    # Split data into features and targets
    X = data[feature_columns]
    y = data[TARGET_VARIABLE]
    
    return X, y, feature_columns

def evaluate_model_performance(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """Evaluate and print model performance metrics."""
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                auc = metrics.roc_auc_score(y_true, y_pred_proba)
                print(f"AUC:       {auc:.4f}")
            else:  # Multi-class
                auc = metrics.roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                print(f"AUC (OvR): {auc:.4f}")
        except ValueError as e:
            print(f"AUC calculation failed: {e}")
    
    return accuracy, precision, recall, f1

def train_deep_neural_network(X_train, X_test, y_train, y_test, feature_columns):
    """Train and evaluate Deep Neural Network (MLPClassifier with multiple hidden layers)."""
    print("\n" + "="*50)
    print("TRAINING DEEP NEURAL NETWORK (MLPClassifier)")
    print("="*50)
    
    start_time = time.time()
    
    # Properly scale features (fit on train, transform both)
    print("Scaling features for DNN...")
    X_train_scaled, scaler = prepare_features(
        pd.DataFrame(X_train, columns=feature_columns), feature_columns, fit_scaler=True
    )
    X_test_scaled, _ = prepare_features(
        pd.DataFrame(X_test, columns=feature_columns), feature_columns, scaler=scaler, fit_scaler=False
    )
    
    # Grid search parameters for deep network (multiple hidden layers)
    param_grid = {
        'hidden_layer_sizes': [
            (128, 64, 32),       # 3 layers
            (256, 128, 64),      # 3 layers (larger)
            (512, 256, 128),     # 3 layers (very wide)
            (256, 128, 64, 32),  # 4 layers
        ],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lgfbs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
    
    print("Performing grid search for deep neural network...")
    clf = neural_network.MLPClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.2)
    
    grid_search = model_selection.GridSearchCV(
        clf, param_grid, 
        scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy',
        cv=3,  # Reduced CV folds for faster training with deep networks
        refit=True, 
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test on holdout set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    end_time = time.time()
    print(f"DNN training completed in {end_time - start_time:.2f} seconds")
    
    # Handle probability output for binary vs multiclass
    if len(np.unique(y_train)) == 2:
        y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
    
    evaluate_model_performance(y_test, y_pred, y_pred_proba, "Deep Neural Network")
    
    return best_model, scaler

def train_shallow_neural_network(X_train, X_test, y_train, y_test, feature_columns):
    """Train and evaluate Shallow Neural Network (single hidden layer MLPClassifier)."""
    print("\n" + "="*50)
    print("TRAINING SHALLOW NEURAL NETWORK (Single Hidden Layer)")
    print("="*50)
    
    start_time = time.time()
    
    # Properly scale features (fit on train, transform both)
    print("Scaling features for SNN...")
    X_train_scaled, scaler = prepare_features(
        pd.DataFrame(X_train, columns=feature_columns), feature_columns, fit_scaler=True
    )
    X_test_scaled, _ = prepare_features(
        pd.DataFrame(X_test, columns=feature_columns), feature_columns, scaler=scaler, fit_scaler=False
    )
    
    # Grid search parameters for shallow network (single hidden layer)
    param_grid = {
        'hidden_layer_sizes': [(50,), (80,), (128,), (256,)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
    }
    
    print("Performing grid search for shallow neural network...")
    clf = neural_network.MLPClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.2)
    
    grid_search = model_selection.GridSearchCV(
        clf, param_grid, 
        scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy',
        cv=5, 
        refit=True, 
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test on holdout set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    end_time = time.time()
    print(f"SNN training completed in {end_time - start_time:.2f} seconds")
    
    # Handle probability output for binary vs multiclass
    if len(np.unique(y_train)) == 2:
        y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
    
    evaluate_model_performance(y_test, y_pred, y_pred_proba, "Shallow Neural Network")
    
    return best_model, scaler

def train_svm(X_train, X_test, y_train, y_test, feature_columns):
    """Train and evaluate SVM with grid search."""
    print("\n" + "="*50)
    print("TRAINING SUPPORT VECTOR MACHINE")
    print("="*50)
    
    start_time = time.time()
    
    # Properly scale features (fit on train, transform both)
    print("Scaling features for SVM...")
    X_train_scaled, scaler = prepare_features(
        pd.DataFrame(X_train, columns=feature_columns), feature_columns, fit_scaler=True
    )
    X_test_scaled, _ = prepare_features(
        pd.DataFrame(X_test, columns=feature_columns), feature_columns, scaler=scaler, fit_scaler=False
    )
    
    # Grid search parameters
    param_grid = {
        'C': [5, 10, 20],
        'gamma': [0.1, 'scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
    }
    
    print("Performing grid search...")
    clf = svm.SVC(probability=True, random_state=SEED)
    
    grid_search = model_selection.GridSearchCV(
        clf, param_grid,
        scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy',
        cv=5,
        refit=True,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test on holdout set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    end_time = time.time()
    print(f"SVM training completed in {end_time - start_time:.2f} seconds")
    
    # Handle probability output for binary vs multiclass
    if len(np.unique(y_train)) == 2:
        y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
    
    evaluate_model_performance(y_test, y_pred, y_pred_proba, "Support Vector Machine")
    
    return best_model


def train_lightgbm(X_train, X_test, y_train, y_test, feature_columns):
    # Due to compatibility issues, this does not utilize GridSearchCV
    print("\n" + "="*50)
    print("TRAINING LIGHTGBM")
    print("="*50)
   
    start_time = time.time()
   
    # LightGBM can handle non-scaled data, but we'll scale for consistency
    print("Scaling features for LightGBM...")
    X_train_scaled, scaler = prepare_features(
        pd.DataFrame(X_train, columns=feature_columns), feature_columns, fit_scaler=True
    )
    X_test_scaled, _ = prepare_features(
        pd.DataFrame(X_test, columns=feature_columns), feature_columns, scaler=scaler, fit_scaler=False
    )
    
    # Convert the training data to LGBM dataset
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    
    # Grid search parameters
    num_leaves_list = [31, 63, 127, 255]
    learning_rate_list = [0.01, 0.05, 0.1]
    num_round_list = [50, 100]
    
    best_auc = -1
    best_params = None
    best_model = None
    
    print("Performing grid search...")
    
    for num_leaves in num_leaves_list:
        for learning_rate in learning_rate_list:
            for num_round in num_round_list:
                
                # Define the LightGBM model parameters
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': num_leaves,
                    'learning_rate': learning_rate,
                    'verbose': -1,
                    'force_col_wise': True
                }
                
                # Train the model
                bst = lgb.train(params, train_data, num_round)
                
                # Make predictions on the test set
                y_pred = bst.predict(X_test_scaled)
                y_pred_binary = (y_pred >= 0.5).astype(int)
                
                # Evaluate the model
                accuracy = metrics.accuracy_score(y_test, y_pred_binary)
                precision = metrics.precision_score(y_test, y_pred_binary)
                auc = metrics.roc_auc_score(y_test, y_pred)
                
                # Track best model
                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'n_estimators': num_round
                    }
                    best_model = bst
                    # print("*** NEW BEST MODEL ***")
                if params["verbose"] > -1:
                    print(f"LGBM with {num_leaves} leaves, {learning_rate} learning rate, and {num_round} rounds.")
                    print("Accuracy: ", accuracy)
                    print("Precision: ", precision)
                    print("AUC: ", auc)
                    print("-" * 30)
    
    end_time = time.time()
    print(f"\nBest parameters: {best_params}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"LightGBM training completed in {end_time - start_time:.2f} seconds")
    
    return best_model

def main():
    """Run Hyperparameter Grid Search and Model Comparison"""
    print("AI Propaganda Classification - Model Comparison")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Topic: {TOPIC}")
    print(f"Features: {X_FEATURES}")
    print(f"Database: {PATH}")
    
    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data()
    
    # Split data into train/test sets (no validation needed for sklearn models)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    
    # Store results
    results = {}
    
    # Train models
    if RUN_DNN:
        try:
            dnn_model = train_deep_neural_network(X_train, X_test, y_train, y_test, feature_columns)
            results['DNN'] = dnn_model
        except Exception as e:
            print(f"Error training DNN: {e}")
    
    if RUN_SNN:
        try:
            snn_model = train_shallow_neural_network(X_train, X_test, y_train, y_test, feature_columns)
            results['SNN'] = snn_model
        except Exception as e:
            print(f"Error training SNN: {e}")
    
    if RUN_SVM:
        try:
            svm_model = train_svm(X_train, X_test, y_train, y_test, feature_columns)
            results['SVM'] = svm_model
        except Exception as e:
            print(f"Error training SVM: {e}")
    
    if RUN_LGBM:
        try:
            lgbm_model  = train_lightgbm(X_train, X_test, y_train, y_test, feature_columns)
            results['LGBM'] = lgbm_model

        except Exception as e:
            print(f"Error training LightGBM: {e}")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON COMPLETE")
    print("="*60)
    
    # Uncomment to save models
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # for model_name, model in results.items():
    #     if model is not None:
    #         try:
    #             # Save model
    #             model_filename = f"{model_name}_{MODEL}_{X_FEATURES}_{timestamp}.joblib"
    #             dump(model, model_filename)
    #             print(f"Saved {model_name} model to {model_filename}")
    #         except Exception as e:
    #             print(f"Failed to save {model_name} model: {e}")

if __name__ == "__main__":
    main()