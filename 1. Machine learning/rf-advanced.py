from google.colab import files
uploaded = files.upload()

import pandas as pd

# Langkah 1: Persiapan Data

# Load dataset
data = pd.read_csv('jira305.csv')

# Display first few rows of the dataset
# Melihat beberapa data untuk verifikasi
data.tail()

# Langkah 2: Preprocessing Data
from sklearn.preprocessing import LabelEncoder

# Mengubah kolom tanggal ke format datetime
data['Created'] = pd.to_datetime(data['Created'])
data['Updated'] = pd.to_datetime(data['Updated'])
data['Resolved'] = pd.to_datetime(data['Resolved'])

# Encoder fitur kategori
priority_encoder = LabelEncoder()
resolution_encoder = LabelEncoder()
data['Priority'] = priority_encoder.fit_transform(data['Priority'])
data['Resolution'] = resolution_encoder.fit_transform(data['Resolution'])

# Konversi kolom tanggal ke numerik (timestamp)
data['Created'] = data['Created'].apply(lambda x: x.timestamp())
data['Updated'] = data['Updated'].apply(lambda x: x.timestamp())
data['Resolved'] = data['Resolved'].apply(lambda x: x.timestamp())

# Memilih fitur dan label
features = ['Priority', 'Resolution', 'Created', 'Updated', 'Resolved']
X = data[features]
y = data['Issue Type'].apply(lambda x: 1 if x == 'Bug' else 0)

# Mengisi nilai NaN jika ada
X = X.fillna(0)

# Split data menjadi train dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Langkah 3: Pelatihan Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Melatih model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediksi pada data test
y_pred = clf.predict(X_test)

# Evaluasi model
print(classification_report(y_test, y_pred))

import joblib

# Save the model
joblib.dump(clf, 'random_forest_bug_predictorfaiq.pkl')

# Tren analisis

import matplotlib.pyplot as plt

# Menghitung jumlah tiket bug dan non-bug per bulan
data['Created_month'] = pd.to_datetime(data['Created'], unit='s').dt.to_period('M')
bug_trend = data[data['Issue Type'] == 'Bug'].groupby('Created_month').size()
non_bug_trend = data[data['Issue Type'] == 'Not Bug'].groupby('Created_month').size()

# Visualisasi tren
plt.figure(figsize=(12, 6))
bug_trend.plot(label='Bug', marker='o')
non_bug_trend.plot(label='Not Bug', marker='x')
plt.title('Tren Laporan Bug dan Non-Bug Per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Tiket')
plt.legend()
plt.show()

# Alokasi Sumber Daya
# Membuat kolom probabilitas prediksi
y_prob = clf.predict_proba(X)[:, 1]
data['Bug Probability'] = y_prob

# Menentukan tiket dengan probabilitas bug tinggi
high_prob_bugs = data[data['Bug Probability'] > 0.7]

# Menampilkan beberapa tiket dengan probabilitas bug tinggi
high_prob_bugs[['Summary', 'Priority', 'Bug Probability']]

# Menghitung kinerja penanganan bug
resolved_bugs = data[(data['Issue Type'] == 'Bug') & (data['Status'] == 'Done')]
avg_resolution_time = (resolved_bugs['Resolved'] - resolved_bugs['Created']).mean()

print(f'Rata-rata waktu penyelesaian bug: {avg_resolution_time} detik')

######################################################################################
##############################IMPLEMENTATION##########################################
######################################################################################

from google.colab import files
uploaded = files.upload()

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Membaca data bug ticket baru
# new_data = pd.read_csv('/mnt/data/new_bug_tickets.csv')
new_data = pd.read_csv('newbugsticket8jan2.csv')

# Ubah kolom tanggal ke datetime
new_data['Created'] = pd.to_datetime(new_data['Created'])
new_data['Updated'] = pd.to_datetime(new_data['Updated'])
new_data['Resolved'] = pd.to_datetime(new_data['Resolved'])

# Pilih fitur yang sama dengan yang digunakan saat melatih model
features = ['Summary', 'Priority', 'Resolution', 'Created', 'Updated', 'Resolved']
X_new = new_data[features]

# Encoder fitur kategori
priority_encoder = LabelEncoder()
resolution_encoder = LabelEncoder()
X_new['Priority'] = priority_encoder.fit_transform(X_new['Priority'])
X_new['Resolution'] = resolution_encoder.fit_transform(X_new['Resolution'])

# Konversi kolom tanggal ke numerik (timestamp)
X_new['Created'] = X_new['Created'].astype(int) / 10**9
X_new['Updated'] = X_new['Updated'].astype(int) / 10**9
X_new['Resolved'] = X_new['Resolved'].astype(int) / 10**9

# Mengisi nilai NaN jika ada
X_new = X_new.fillna(0)

# Load model yang telah dilatih
# clf = joblib.load('/path_to_trained_model.joblib')
clf = joblib.load('random_forest_bug_predictorfaiq.pkl')

# Sesuaikan kolom data baru dengan data pelatihan (menggunakan fitur yang digunakan saat melatih model)
X_new = X_new.reindex(columns=clf.feature_names_in_, fill_value=0)

# Membuat prediksi pada data baru
y_new_pred = clf.predict(X_new)

# Menambahkan kolom prediksi ke data bug ticket baru
new_data['Predicted Issue Type'] = y_new_pred

# Mengonversi label prediksi ke bentuk aslinya
new_data['Predicted Issue Type'] = new_data['Predicted Issue Type'].apply(lambda x: 'Bug' if x == 1 else 'Not Bug')

# Menyimpan hasil prediksi ke file baru
# new_data.to_csv('/mnt/data/predicted_bug_tickets.csv', index=False)

# Menampilkan hasil prediksi
print(new_data[['Summary', 'Predicted Issue Type']])
