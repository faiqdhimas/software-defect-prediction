from google.colab import files
uploaded = files.upload()

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Read dataset
data = pd.read_csv('jira305.csv')

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown', inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Convert text features to binary indicators
data['Description'] = data['Description'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Custom field (Step to Reproduce)'] = data['Custom field (Step to Reproduce)'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Comment'] = data['Comment'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)

# Encode categorical features
categorical_columns = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Convert date columns to numeric
date_columns = ['Created', 'Updated', 'Resolved']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    data[col].fillna(data[col].median(), inplace=True)
    data[col] = data[col].astype(int) / 10**9

# Convert 'Issue Type' to binary: 1 for 'Bug', 0 for 'Not Bug'
data['Issue Type'] = data['Issue Type'].apply(lambda x: 1 if x == 'Bug' else 0)

# Select features and target
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
X = data[selected_features]
y = data['Issue Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print feature information before Bat Algorithm
print(f"Total features before Bat Algorithm: {X_train.shape[1]}")

# Train SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f'Akurasi Model: {accuracy:.2f}')
print('Classification Report:')
print(report)
print(f'AUC-ROC: {auc:.2f}')
print('Confusion Matrix:')
print(cm)

# Save model and preprocessing tools
joblib.dump(svm, 'svm_model2.pkl')
joblib.dump(scaler, 'scaler.pkl')
for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')

import matplotlib.pyplot as plt
import seaborn as sns

# Melihat kepentingan fitur
importance = svm.coef_[0]
features = list(X.columns)
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
print(importance_df.sort_values(by='Importance', ascending=False))

# Menyimpan dataframe ke dalam file CSV
importance_df.to_csv('feature_importance.csv', index=False)

# Menyimpan visualisasi ke dalam file gambar
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()

######################################################################################
##############################IMPLEMENTATION##########################################
######################################################################################

from google.colab import files
uploaded = files.upload()

import pandas as pd
import joblib

# Load the trained SVM model and preprocessing tools
svm_model = joblib.load('svm_model2.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders_new = {col: joblib.load(f'label_encoder_{col}.pkl') for col in ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Resolution', 'Assignee', 'Reporter', 'Creator', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary']}

# Read new data
# new_data = pd.read_csv('/mnt/data/jun9-testdataset40an(withresoo).csv')
new_data = pd.read_csv('jun9-testdataset40an(withreso).csv')

# Handle missing values in new data
for col in new_data.columns:
    if new_data[col].dtype == 'object':
        new_data[col].fillna(new_data[col].mode()[0] if not new_data[col].mode().empty else 'unknown', inplace=True)
    else:
        new_data[col].fillna(new_data[col].median(), inplace=True)

# Apply label encoding using the loaded encoders, with handling for unseen labels
for col in label_encoders_new:
    le = label_encoders_new[col]
    new_data[col] = new_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Convert date columns to numeric
date_columns = ['Created', 'Updated', 'Resolved']
for col in date_columns:
    new_data[col] = pd.to_datetime(new_data[col], errors='coerce')
    new_data[col].fillna(new_data[col].median(), inplace=True)
    new_data[col] = new_data[col].astype(int) / 10**9

# Select relevant features for classification
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Resolution', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary']
X_new = new_data[selected_features]

# Normalize features
X_new_scaled = scaler.transform(X_new)

# Predict using the trained SVM model
predictions = svm_model.predict(X_new_scaled)

# Print predictions
for i, prediction in enumerate(predictions):
    print(f"Ticket {i+1}: This ticket likely {'contains' if prediction == 1 else 'does not contain'} a bug.")
