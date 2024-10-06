from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle

class ModelTraining:
    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'DecisionTreeRegressor':DecisionTreeRegressor(),
            'XGBRegressor':XGBRegressor()
        }
        self.best_model = None

    def train_test_split(self, test_size=0.2, random_state=42):
        
       
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, metric='rmse'):
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if metric == 'rmse':
            score = mean_squared_error(y_test, predictions, squared=False)  # RMSE
        elif metric == 'r2':
            score = r2_score(y_test, predictions)  # R-squared score
        else:
            raise ValueError("Unsupported metric. Use 'rmse' or 'r2'.")
        
        return score

    def compare_models(self, metric='rmse'):
        
        X_train, X_test, y_train, y_test = self.train_test_split()
        model_scores = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            score = self.evaluate_model(model, X_train, X_test, y_train, y_test, metric)
            model_scores[model_name] = score
            print(f"{model_name} {metric.upper()}: {score:.4f}")

        # Select the best model
        if metric == 'rmse':
            self.best_model_name= min(model_scores, key=model_scores.get)  # Lower RMSE is better
        elif metric == 'r2':
            self.best_model_name = max(model_scores, key=model_scores.get)  # Higher R2 is better
        
        self.best_model = self.models[self.best_model_name]

        print(f"\nBest Model: {self.best_model_name} with {metric.upper()}: {model_scores[self.best_model_name]:.4f}")
        return self.best_model

    def train_best_model(self, metric='r2'):
        
        best_model = self.compare_models(metric)
        X_train, X_test, y_train, y_test = self.train_test_split()
        best_model.fit(X_train, y_train)
        return best_model
    
    def save_best_model(self, filename='best_model.pkl'):
        if self.best_model:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Best model ({self.best_model_name}) saved to {filename}")
        else:
            print("No model found. Train the model first before saving.")
