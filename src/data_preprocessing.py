import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '../data/eeg_eye.csv'
data = pd.read_csv(file_path)

# Check for missing values
print(data.isnull().sum())

# Separate features and target
X = data.drop(columns='eyeDetection')
y = data['eyeDetection']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
