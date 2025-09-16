from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_registry() -> Dict[str, Any]:
    return {
        'random_forest': RandomForestRegressor(n_estimators=600, max_depth=None, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'svr': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=10.0, epsilon=0.05, kernel='rbf')),
        ]),
    }
