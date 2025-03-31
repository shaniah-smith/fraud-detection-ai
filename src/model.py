import pandas as pd
from sklearn.ensemble import IsolationForest

def train_model(file_path):
    df = pd.read_csv(file_path)

    if 'is_fraud' in df.columns:
        features = df.drop(columns=['is_fraud'])
    else:
        features = df.copy()

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)

    predictions = model.predict(features)
    return model, predictions