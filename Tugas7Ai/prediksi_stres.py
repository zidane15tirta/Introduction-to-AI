import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('data_stress.csv', sep=';')

# ============================================================
# 2. CLEANING
# ============================================================
df_clean = df.iloc[:, 4:11].dropna()
X = df_clean.iloc[:, 0:6]
y = df_clean.iloc[:, 6]

# ============================================================
# 3. ENCODING (simpan encoder tiap kolom)
# ============================================================
le_dict = {}
X_encoded = X.copy()
for col in X.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# ============================================================
# 4. TRAINING (80:20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)

# ============================================================
# 5. EVALUASI MODEL
# ============================================================
y_pred = model.predict(X_test)
print("=" * 60)
print("        LAPORAN EVALUASI MODEL NAIVE BAYES")
print("=" * 60)
print(f"Akurasi Model : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("-" * 60)
print("DETAIL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))
print("-" * 60)

# ============================================================
# 6. PREDIKSI DATA BARU
# Pilihan nilai tiap kolom:
#   Semester saat ini     : 'Semester 1–2' | 'Semester 3–4' | 'Semester 5–6'
#   Jam tidur per hari    : '< 5 jam'      | '5–6 jam'      | '7–8 jam'
#   Jumlah tugas/minggu   : '3–4'          | '5–6'          | '> 6'
#   Sering kelelahan      : 'Kadang'        | 'Sering'
#   Sering cemas          : 'Kadang'        | 'Sering'
#   Ikut organisasi       : 'Ya'            | 'Tidak'
# ============================================================
input_data = {
    'Semester saat ini '                                          : 'Semester 3–4',
    'Berapa jam tidur rata-rata Anda per hari?  '                 : '< 5 jam',
    'Berapa jumlah tugas yang Anda kerjakan dalam 1 minggu?  '    : '> 6',
    'Seberapa sering Anda merasa kelelahan? '                     : 'Sering',
    'Seberapa sering Anda merasa cemas terkait kuliah/tugas? '    : 'Sering',
    'Apakah Anda mengikuti organisasi/kegiatan kampus? '          : 'Ya',
}

# Validasi & encode input
input_df = pd.DataFrame([input_data])
input_encoded = input_df.copy()
valid = True

for col in input_df.columns:
    val = input_df[col].astype(str).values[0]
    valid_labels = list(le_dict[col].classes_)
    if val not in valid_labels:
        print(f"[ERROR] Kolom '{col}': nilai '{val}' tidak valid!")
        print(f"        Nilai yang tersedia: {valid_labels}")
        valid = False
    else:
        input_encoded[col] = le_dict[col].transform(input_df[col].astype(str))

if valid:
    hasil = model.predict(input_encoded)
    print("        HASIL PREDIKSI DATA MAHASISWA")
    print("-" * 60)
    for col, val in input_data.items():
        print(f"  {col.strip():<45} : {val}")
    print("-" * 60)
    print(f"  Kesimpulan AI : Mahasiswa ini memiliki Tingkat Stres → {hasil[0]}")
    print("=" * 60)