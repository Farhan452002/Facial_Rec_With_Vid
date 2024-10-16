import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
df = pd.read_csv('winequalityN.csv')
print("Initial DataFrame:")
print(df.head())
# Replace categorical values with numerical values
df['type'].replace(['white', 'red'],
                        [0, 1], inplace=True)

print("\nDataFrame after encoding categorical values:")
print(df.head())

# Check for null values and replace with the column's mean
print("\nChecking for null values:")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

# Check if null values are handled
print("\nNull values after filling:")
print(df.isnull().sum())

# Normalize numerical columns if they have a wide range of values
numerical_cols = df.select_dtypes(include=[np.number]).columns
print("\nNumerical columns before normalization:")
print(df[numerical_cols].describe())

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nNumerical columns after normalization:")
print(df[numerical_cols].describe())

df_corr = df.drop(columns=['quality'])

# Remove highly correlated columns
corr_matrix = df.corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# Create a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Find columns with correlation greater than 0.7
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

print("\nColumns to drop due to high correlation:")
print(to_drop)

df.drop(columns=to_drop, inplace=True)

print("\nDataFrame after dropping highly correlated columns:")
print(df.head())

# Bin the 'quality' column into categories
bins = [0, 4, 6, 10]  # Define bin edges
labels = ['low', 'medium', 'high']  # Define labels for the bins
df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)

# Encode the binned quality labels
le_quality = LabelEncoder()
df['quality'] = le_quality.fit_transform(df['quality'])

# Prepare the data for modeling
X = df.drop(columns=['quality', 'type'])
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate k-Nearest Neighbors model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("k-Nearest Neighbors Classifier:")
print(classification_report(y_test, y_pred_knn))
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn)}")

# Train and evaluate Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt)}")

# Train and evaluate Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")

'''
Output:
k-Nearest Neighbors Classifier:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1298
           1       0.00      0.00      0.00         2

    accuracy                           1.00      1300
   macro avg       0.50      0.50      0.50      1300
weighted avg       1.00      1.00      1.00      1300

Accuracy: 0.9984615384615385

Decision Tree Classifier:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1298
           1       0.00      0.00      0.00         2

    accuracy                           1.00      1300
   macro avg       0.50      0.50      0.50      1300
weighted avg       1.00      1.00      1.00      1300

Accuracy: 0.9953846153846154

Random Forest Classifier:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1298
           1       0.00      0.00      0.00         2

    accuracy                           1.00      1300
   macro avg       0.50      0.50      0.50      1300
weighted avg       1.00      1.00      1.00      1300

Accuracy: 0.9984615384615385
'''
