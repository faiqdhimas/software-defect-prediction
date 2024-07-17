import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import lightgbm as lgb

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

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    joblib.dump(le, f'label_encoder_{col}.pkl')

# Convert date columns to numeric
date_columns = ['Created', 'Updated', 'Resolved']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    data[col].fillna(data[col].median(), inplace=True)
    data[col] = data[col].astype(int) / 10**9

# Select features and target
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
X = data[selected_features]
y = data['Issue Type'].apply(lambda x: 1 if x == 'Bug' else 0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Train LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Save model
joblib.dump(lgb_model, 'lgb_model.pkl')

# Evaluate model
y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f'Akurasi Model: {accuracy:.2f}')
print('Classification Report:')
print(report)
print(f'AUC-ROC: {auc:.2f}')
print('Confusion Matrix:')
print(cm)


######################################################################################
##############################IMPLEMENTATION##########################################
######################################################################################


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Read new dataset (without 'Issue Type')
new_data = pd.read_csv('jira43tests.csv')

# Handle missing values
for col in new_data.columns:
    if new_data[col].dtype == 'object':
        new_data[col].fillna(new_data[col].mode()[0] if not new_data[col].mode().empty else 'unknown', inplace=True)
    else:
        new_data[col].fillna(new_data[col].median(), inplace=True)

# Encode categorical features with handling unseen labels
categorical_columns = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']

for col in categorical_columns:
    le = joblib.load(f'label_encoder_{col}.pkl')
    new_data[col] = new_data[col].apply(lambda x: x if x in le.classes_ else 'unknown')
    le.classes_ = np.append(le.classes_, 'unknown')
    new_data[col] = le.transform(new_data[col])

# Convert date columns to numeric
date_columns = ['Created', 'Updated', 'Resolved']
for col in date_columns:
    new_data[col] = pd.to_datetime(new_data[col], errors='coerce')
    new_data[col].fillna(new_data[col].median(), inplace=True)
    new_data[col] = new_data[col].astype(int) / 10**9

# Select features
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
X_new = new_data[selected_features]

# Standardize features
scaler = joblib.load('scaler.pkl')
X_new = scaler.transform(X_new)

# Load model
lgb_model = joblib.load('lgb_model.pkl')

# Predict issue type
y_pred = lgb_model.predict(X_new)
y_pred_prob = lgb_model.predict_proba(X_new)[:, 1]

# Append predictions to the new data
new_data['Issue Type'] = y_pred
new_data['Issue Type'] = new_data['Issue Type'].apply(lambda x: 'Bug' if x == 1 else 'Not Bug')
new_data['Issue Type Probability'] = y_pred_prob

# Print the results
for index, row in new_data.iterrows():
    print(f"Ticket {index}: Issue Type = {row['Issue Type']}, Probability = {row['Issue Type Probability']:.2f}")

print("Predictions have been printed.")
