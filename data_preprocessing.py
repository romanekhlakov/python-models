import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)
    
    # Преобразование столбца 'Time' в datetime формат
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Создание дополнительных признаков
    data['Hour'] = data['Time'].dt.hour
    data['DayOfWeek'] = data['Time'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'] >= 5
    data['Month'] = data['Time'].dt.month
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
    # Удаление временной метки для модели
    data = data.drop(columns=['Time'])
    
    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Hour', 'DayOfWeek', 'IsWeekend', 'Month', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos']] = scaler.fit_transform(data[['Hour', 'DayOfWeek', 'IsWeekend', 'Month', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos']])
    data['Average_Speed_kmph'] = target_scaler.fit_transform(data['Average_Speed_kmph'].values.reshape(-1, 1))
    
    return data, scaler, target_scaler

def create_dataset(data, window_size=12):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size].values)
        y.append(data['Average_Speed_kmph'].iloc[i + window_size])
    return np.array(X), np.array(y)
