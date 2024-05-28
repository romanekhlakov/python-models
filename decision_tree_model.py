from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data, create_dataset
from tqdm import tqdm

def decision_tree_model(file_path, window_size=72):
    data, scaler, target_scaler = load_and_preprocess_data(file_path)
    X, y = create_dataset(data, window_size)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = DecisionTreeRegressor()
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
    
    print("Training Decision Tree model with Grid Search...")
    for _ in range(1):
        grid_search.fit(X_train_flat, y_train)
    
    print(f'Best parameters: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_flat)
    
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    return y_true, y_pred
