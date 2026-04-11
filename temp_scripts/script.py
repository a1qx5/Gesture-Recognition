import pandas as pd

data = pd.read_csv('../data/gestures_data.csv')
data = data[data['gesture_id'] != 8]

data.to_csv('../data/gestures_data.csv')