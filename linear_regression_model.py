from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from data_preprocessing import load_and_preprocess_data, create_dataset
from tqdm import tqdm

def linear_regression_model(file_path, window_size=72):
    data, scaler, target_scaler = load_and_preprocess_data(file_path)
    X, y = create_dataset(data, window_size)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = LinearRegression()
    scores = cross_val_score(model, X_train_flat, y_train, cv=5, scoring='r2')
    print(f'Cross-validation R^2 scores: {scores}')
    
    print("Training Linear Regression model...")
    for _ in tqdm(range(1)):
        model.fit(X_train_flat, y_train)
    
    y_pred = model.predict(X_test_flat)
    
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    return y_true, y_pred
