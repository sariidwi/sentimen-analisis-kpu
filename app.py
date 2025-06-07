from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from google_play_scraper import reviews, Sort
from datetime import datetime
import pickle
import csv
import os
import re
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import base64
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = 'rahasia'

# --- Upload folder setup ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MySQL Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/sirekap_sentimen'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()



# --- DB Model ---
class Ulasan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.String(255))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
    input_source = db.Column(db.Enum('manual', 'scraping'), nullable=False, default='manual')
    original_filename = db.Column(db.String(255), nullable=True)
    source_url = db.Column(db.String(500), nullable=True)
    preprocessed = db.relationship('PreprosesData', backref='ulasan', lazy=True)
    
    
class PreprosesData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ulasan_id = db.Column(db.Integer, db.ForeignKey('ulasan.id'))
    original = db.Column(db.Text, nullable=False)
    cleaned = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

with app.app_context():
    db.create_all()

# --- Load Model ML ---
model_path = os.path.join(os.path.dirname(__file__), 'modelssssss', 'sentiment_models.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'modelssssss', 'vectorizerr.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)



# --- Login Config ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy user dengan hashed password
from werkzeug.security import generate_password_hash, check_password_hash
users = {'admin': {'password': generate_password_hash('admin123')}}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# =========================
# === ROUTES START HERE ===
# =========================

# --- Login ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = users.get(username)

        if user_data and check_password_hash(user_data['password'], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Username atau password salah.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Home ---
@app.route('/home')
@login_required
def home():
    return render_template('home.html', title="Beranda")

# --- Input Data Page (menu utama) ---
@app.route('/input-data')
@login_required
def input_data():
    return render_template('input_data.html', title="Input Data")

# --- Manual Input CSV ---
@app.route('/input-manual', methods=['GET', 'POST'])
@login_required
def manual_input():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file.filename.endswith('.csv'):
            flash('Hanya file CSV yang diperbolehkan.')
            return redirect(request.url)
        if not file or file.filename == '':
            flash('File tidak ditemukan atau kosong.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # === Tambahkan log debug di sini ===
        print("File berhasil diunggah:", filename)
        print("Lokasi penyimpanan:", filepath)
        comments = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # aman kalau kosong
                print("Header CSV:", header)

                for row in reader:
                    print("Row dibaca:", row)
                    if not row or len(row) < 1:
                        continue
                    content = row[0].strip()
                    if not content:
                        continue
                    ulasan = Ulasan(
                        content=content,
                        input_source='manual',
                        original_filename=filename,
                        timestamp=datetime.now()
                    )
                    db.session.add(ulasan)
                    comments.append(content)
            db.session.commit()
            return render_template('input_manual_result.html', comments=comments, title="Input Manual", csv_file=filename)
        except Exception as e:
            import traceback
            traceback.print_exc()  # cetak error di terminal Flask
            flash(f"Terjadi kesalahan saat membaca file: {e}")
            return redirect(request.url)

    return render_template('input_manual.html', title="Input Manual")

# --- Scraping dari Playstore ---
@app.route('/scraping', methods=['GET', 'POST'])
@login_required
def scrape_input():
    if request.method == 'POST':
        package = request.form['package']
        count = int(request.form.get('jumlah', 100))  # default 100 jika tidak ada input

        if not package.strip():
            flash("Package ID tidak boleh kosong.")
            return redirect(request.url)

        result, _ = reviews(
            package,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=count
        )
        # Simpan ke CSV dan Database
        filename = f"scraped_reviews_{package.replace('.', '_')}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        comments = []

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['app_id', 'review', 'datetime'])

            for r in result:
                content = r['content']
                at = r['at']  # ini datetime
                writer.writerow([package, content, at.strftime("%Y-%m-%d %H:%M:%S")])
                ulasan = Ulasan(app_id=package, content=content, timestamp=at, input_source='scraping')
                db.session.add(ulasan)
                comments.append((content, at.strftime("%d-%m-%Y %H:%M")))
        db.session.commit()

        return render_template('scraping_result.html', comments=comments, title="Scraping", csv_file=filename)

    return render_template('scraping_form.html', title="Scraping Playstore")

# --- Insight Halaman ---
@app.route('/insight')
@login_required
def insight():
    return render_template('insight.html', title="Insight")

@app.route('/', methods=['GET', 'POST'])
@login_required
def predict():
    sentiment = None
    review = ""

    if request.method == "POST":
        review = request.form["ulasan"]
        if not review.strip():
            flash("Ulasan tidak boleh kosong.")
            return redirect(request.url)

        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == "Positif":
            sentiment = "Positif 🙂"
        elif prediction == "Negatif":
            sentiment = "Negatif 😠"
        else:
            sentiment = "Netral 😐"

    return render_template("predict.html", sentiment=sentiment, review=review, title="Prediksi Ulasan")

# Contoh fungsi preprocessing (bisa Anda sesuaikan lebih lanjut)
def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'https?://\S+|www\.\S+', '', teks)  # hapus URL
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)  # hapus simbol
    teks = re.sub(r'\s+', ' ', teks).strip()  # hapus spasi berlebih
    return teks

# pelabelan menggunakan ML
def label_ml(teks):
    if model is None or vectorizer is None:
        raise ValueError("Model atau vectorizer belum dimuat.")

    if not teks.strip():
        return 'Netral'

    teks_bersih = bersihkan_teks(teks)
    vektor = vectorizer.transform([teks_bersih])
    prediksi = model.predict(vektor)[0]
    
    return prediksi


# --- Preprocesing --- 
@app.route('/preprocessing-labeling', methods=['GET', 'POST'])
@login_required
def preprocessing():
    hasil = []

    if request.method == 'POST':
        ulasan_list = Ulasan.query.order_by(Ulasan.timestamp.desc()).all()

        for ulasan in ulasan_list:
            # Cek apakah ulasan sudah diproses
            existing = PreprosesData.query.filter_by(ulasan_id=ulasan.id).first()
            if existing:
                continue
    
            original = ulasan.content
            cleaned = bersihkan_teks(original)
            label = label_ml(original)

            data = PreprosesData(
                ulasan_id=ulasan.id,
                original=original,
                cleaned=cleaned,
                label=str(label),
                timestamp=datetime.now()
            )
            db.session.add(data)
            hasil.append({
                'original': original,
                'cleaned': cleaned,
                'label': label,
                'timestamp': ulasan.timestamp.strftime("%d-%m-%Y %H:%M") if ulasan.timestamp else "-"
            })

        db.session.commit()

    return render_template(
        'preprocessing_labeling.html',
        title='Preprocessing dan Labeling Data',
        hasil=hasil
    )


# --- Naive Bayes --- 
@app.route('/naive-bayes', methods=['GET', 'POST'])
@login_required
def naive_bayes_info():
    metrics = None
    data= []

    if request.method == 'POST':
        # Ambil semua data yang sudah dilabeli
        data = PreprosesData.query.all()
        if not data:
            flash("Belum ada data yang diproses.")
            return redirect(request.url)

        texts = [d.cleaned for d in data]
        true_labels = [d.label for d in data]

        # Preprocessing + Vectorizing
        cleaned_texts = [bersihkan_teks(t) for t in texts]
        X_test = vectorizer.transform(cleaned_texts)
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(true_labels, y_pred),
            'precision': precision_score(true_labels, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(true_labels, y_pred, average='weighted', zero_division=0),
            'conf_matrix': confusion_matrix(true_labels, y_pred).tolist(),  # Convert for JSON
            'labels': sorted(list(set(true_labels)))
        }
    return render_template('Naivebayes.html', title='Evaluasi Naive Bayes', metrics=metrics, data=enumerate(data))

# --- Visualisasi --- 
@app.route('/visualisasi')
@login_required
def visualisasi():
    sumber_data = db.session.query(Ulasan.input_source, db.func.count(Ulasan.id)).group_by(Ulasan.input_source).all()
    source_dict = {sumber: count for sumber, count in sumber_data}

    sentiment_counter = {'Positif': 0, 'Negatif': 0, 'Netral': 0}
    all_ulasan = Ulasan.query.all()
    for ulasan in all_ulasan:
        cleaned = bersihkan_teks(ulasan.content)
        label = label_ml(cleaned)  # pakai model ML
        sentiment_counter[label] += 1

    return render_template('Visualisasi.html', title="Visualisasi", source_data=source_dict, sentiment_data=sentiment_counter)

# --- WOrldCloud
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Tambahkan ini agar tidak pakai GUI
import matplotlib.pyplot as plt
import io
import base64

@app.route('/wordcloud')
@login_required
def wordcloud():
    # Ambil data ulasan dari tabel PreprosesData
    data = db.session.query(PreprosesData.cleaned).all()
    all_text = ' '.join([row[0] for row in data if row[0]])

    # Buat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    # Simpan gambar ke buffer
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)

    # Konversi ke base64 agar bisa ditampilkan langsung di HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('wordcloud.html', title="Word Cloud", wordcloud_image=img_base64)

# --- Visualisasi --- 
@app.route('/periode')
@login_required
def periode():
    return render_template('periode.html', title="Periode")


# =========================
# === END OF ROUTES =======
# =========================

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)