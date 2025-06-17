from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from google_play_scraper import reviews, Sort, app as get_app_detail
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
from flask import Response




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
    reviewId = 	db.Column(db.String(255))
    userName = 	db.Column(db.Text)
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
model_path = os.path.join(os.path.dirname(__file__), 'MODELFIX', 'model_final_gabungan.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'MODELFIX', 'vectorizer_final.pkl')

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
    return render_template('Input_data.html', title="Input Data")

@app.route('/hapus_data', methods=['POST'])
@login_required
def hapus_data():
    try:
        # Hapus dulu data anak (PreprosesData)
        db.session.query(PreprosesData).delete()
        db.session.query(Ulasan).delete()
        db.session.commit()
        flash('Semua data berhasil dihapus.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus data: {str(e)}', 'danger')
    return redirect(url_for('input_data'))

# --- Manual Input CSV ---
@app.route('/input-manual', methods=['GET', 'POST'])
@login_required
def manual_input():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('File tidak ditemukan atau kosong.', 'danger')
            return redirect(request.url)

        if not file.filename.endswith('.csv'):
            flash('Hanya file CSV yang diperbolehkan.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print("File berhasil diunggah:", filename)
        print("Lokasi penyimpanan:", filepath)

        comments = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
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

                    comments.append({
                        'content': content,
                        'timestamp': datetime.now().strftime("%d-%m-%Y %H:%M")
                    })

            db.session.commit()

            return render_template(
                'input_manual_result.html',  # Gunakan template yang sama
                comments=comments,
                title="Hasil Input Manual",
                csv_file=filename
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Terjadi kesalahan saat membaca file: {e}", 'danger')
            return redirect(request.url)

    return render_template('input_manual.html', title="Input Manual")

# --- Scraping dari Playstore ---
@app.route('/scraping', methods=['GET', 'POST'])
@login_required
def scrape_input():
    if request.method == 'POST':
        package = request.form['package']
        count = int(request.form.get('jumlah', 100))  # default 100

        if not package.strip():
            flash("Package ID tidak boleh kosong.")
            return redirect(request.url)

        # Cek apakah package ID valid
        try:
            get_app_detail(package)
        except Exception:
            flash("Aplikasi tidak ditemukan di Play Store. Pastikan package ID benar.", 'danger')
            return redirect(request.url)

        # Ambil lebih banyak data, supaya bisa disaring
        fetch_count = count * 3

        result, _ = reviews(
            package,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=fetch_count
        )

        filename = f"scraped_reviews_{package.replace('.', '_')}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        comments = []
        saved_count = 0

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['app_id', 'reviewId', 'userName', 'review', 'datetime'])

            for r in result:
                if saved_count >= count:
                    break

                content = r['content'].strip()
                at = r['at']
                review_id = r.get('reviewId')
                user_name = r.get('userName', 'Anonim')

                # Cek apakah sudah ada
                existing_review = Ulasan.query.filter_by(reviewId=review_id).first() if review_id \
                                else Ulasan.query.filter_by(content=content, timestamp=at).first()
                if existing_review:
                    continue

                # Simpan ke CSV
                writer.writerow([
                    package,
                    review_id,
                    user_name,
                    content,
                    at.strftime("%Y-%m-%d %H:%M:%S")
                ])

                # Simpan ke database
                ulasan = Ulasan(
                    app_id=package,
                    reviewId=review_id,
                    userName=user_name,
                    content=content,
                    timestamp=at,
                    input_source='scraping'
                )
                db.session.add(ulasan)

                # Simpan ke list tampilan
                comments.append({
                    'app_id': package,
                    'reviewId': review_id,
                    'userName': user_name,
                    'content': content,
                    'timestamp': at.strftime("%d-%m-%Y %H:%M")
                })
                
                saved_count += 1

        db.session.commit()

        if not comments:
            flash("Tidak ada ulasan baru yang berhasil diambil.", "warning")

        return render_template('scraping_result.html', comments=comments, title="Hasil Scraping", csv_file=filename)

    return render_template('scraping_form.html', title="Scraping Playstore")

# --- Insight Halaman ---
@app.route('/insight')
@login_required
def insight():
    return render_template('insight.html', title="Wawasan")

# --- predict----
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
import re

# === 1. Stopwords Indonesia (bebas duplikat) ===
stopwords_indonesia = set([
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah", "ini", "itu",
    "sudah", "akan", "telah", "karena", "jika", "seperti", "agar", "atau", "oleh", "dalam",
    "saat", "kami", "kita", "mereka", "saya", "anda", "dia", "hanya", "bukan", "ya",
    "lagi", "lah", "pun", "apa", "berapa", "siapa", "mengapa", "bagaimana", "semua", "setiap",
    "keluar", "klik", "tersebut", "saja", "nya", "terjadi", "sehingga", "baru", "lebih",
    "harus", "yg", "aku", "gue", "ku", "mu", "nih", "deh", "kok", "dong", "udah", "nggak",
    "ngga", "banget", "gitu", "ntar", "gak", "tuh", "si", "eh", "makanya", "malah", "toh",
    "ntah", "yah", "oh", "wkwk", "wkwkwk", "haha", "huhu", "hehe", "hadeh", "astaga", "waduh",
    "loh", "aduh", "ampun", "maaf", "tolong", "please", "makasih", "terima", "kasih", "ok",
    "oke", "sip", "yaudah", "okey", "yok", "ayo", "ayok", "iyaa", "iya", "no", "yes", "thanks",
    "test", "coba", "tes", "uh", "uhm", "hmm", "kenapa", "tapi", "bahkan", "mungkin", "entah", "kalau", 
    "tp", "klo", "karna", "bbrpa", "urusanmu", "jadi", "juga", "mah", "aja",  "seperti", "masih", 
    "pada", "mau", "mana", "lamuuuuu", "hehehe", "eh", " ", "kayak", "kaya", "lah"
     "klo", "ky", "gw", "aja", "gini", "gk", "ga", "g", "ny", "mah", "ne", "donwlod",
    "dong", "loh", "lah", "wah", "aja", "nih", "deh", "yok", "josss", "🤣", "😅", "😭", "wak",
    "🙏", "✌️", "😝", "🔥", "😋", "ampas", "konttoll", "rungkad", "jelekk", "gbk", "sialan", 
    "buosokkk", "josss", "mau", "ya", "iya", "pemirsa", "h", "v", "no", "yes", "wkwk", 
    "wkwkwk", "haha", "huhu", "hehe", "anjay", "gak", "nyusahi", "nyusahke", "rausah", 
    "durung", "nek", "kok", "tok", "aja", "lagi", "gitu", "begitu", "gimana", "kayak",
    "seperti", "udah", "nggak", "ngga", "banget", "cuman", "cumann", "aja", "kayanya", "masak", "kaya", "kayak", "gitu", "dek", "donk",
    "ga", "gk", "g", "gajelas", "gw", "loh", "mah", "nih", "wkwk", "wkwkwk", "hmm",
    "yah", "wkwkw", "deh", "padahal", "ok", "oke", "okey", "sensor"
    "banget", "bikin", "masih", "blm", "udah", "dlu", "dulu", "aja", "siang", "malem",
    "malam", "ny", "jgn", "aja", "lagi", "terimakasih", "biar", "jika", "trus", "yang", 
    "bahkan", "sampai", "sudah", "karna", "karena", "klo", "klw", "klu", "bgt", "bikin", "lakon opo to yooo yooo....",
    "jd", "udh", "tu", "itu", "ini", "dgn", "tp", "dr", "buat", "mau", "toh", "nya", "pdhal", "anggaran", 
    "pun", "foto", "gambar", "hasil", "antara", "mulai", "instruksi", "real", "pdf", "layar", "upload", "online", "ofline", 
    "membagongkan", "gara2", "klw", "Begi", "pea", "ODGJ" "paslon", "02", "SD", "ppwp", "tiba tiba", "kl", "developernya", 
    "selama", "ngapain", "dr", "jd", "yg", "gw", "ga", "Tataian", "iqba", "hpnya", 
    "udh", "dah", "nih", "yaa", "klo", "gk", "nya", "mah", "lg", "tp", "sy", "ya", "edit", 
    "iqba", "terimakasih", "besoknya", "sat", "settt", "adeuh", "teruss", "terussss", "payahhh", 
    "mk", "mxsxx", "dibales", "kak", "via", "bs", "sm", "lho", "gaes", "hbs", "n", "sblmnya", 
    "mdh2n", "hpnya", "maap", "numpang", "disuru", "wkwkwkk", "mh", "anehh", "aslii", 
    "lainn", "khah", "bapakibu", "55", "tuu", "parahhhh", "dg", "ceritanya", "like", "nice", 
    "bang", "the", "valid", "link", "rupiah", "entar", "semacamnya", "salam", "hormat", 
    "duit", "negera", "dikitlah", "budget", "potong", "not", "font", "helvetica", "encoding", 
    "winansiencoding", "clurut", "sisi", "produk2", "wow", "biasalah", "teh", "publish", 
    "real", "jelasss", "uda", "situ", "tahap", "lanjutan", "slow", "sore", "malem", "jelang", 
    "mental", "sehari", "maruk", "kurir", "paswornya", "seharga", "5jt", "hadehhh", "todak", 
    "yak", "kerasnya", "nmr", "telpon", "profilnya", "gada", "aq", "okk", "kendal", "pulak", 
    "pr", "dpt", "tuk", "malh", "token", "sja", "tuntut", "wae", "lier", "pelis", "atuh", 
    "penyakit", "ajaserius", "napaini", "ngelawak", "duluan", "lawas", "main", "aje", 
    "elit", "mff", "ibubapak", "inimah", "malu", "mulus", "rusuh", "acara", "ayo", "masee", 
    "9", "nov", "materi", "donlot", "tapiga", "judule", "lhaaa", "piye", "suda", "hadeeh", 
    "tenaga", "beusaha", "siangmalam", "subu", "kenap", "sya", "masi", "persi", "boskuini", 
    "kayaknya", "playstore", "255", "halo", "amana", "barat", "bisamalah", "apaa", "la", 
    "tidor", "v23", "5g", "bersedia", "ttp", "z", "useless", "ijinkan", "iphone", "drmana", 
    "asalnya", "kompeten", "lu", "kabarin", "dehh", "mntap", "glk", "hdeh", "ssu", "maghrib", 
    "jejeni", "njirlah", "leleeeeeeeeeet", "petot", "sa", "gj", "bluk"
])

# === 2. Kamus Normalisasi (bebas duplikat dan konsisten) ===
norm = {
    "apk": "aplikasi", "apl": "aplikasi", "apknya": "aplikasinya", "aplikasinya": "aplikasi",
    "ap": "apa", "yg": "yang", "skrg": "sekarang", "gak": "tidak", "gk": "tidak", "ga": "tidak", "ngga": "tidak", 
    "ngak": "tidak", "tp": "tapi", "g": "tidak", "gx": "tidak", "gajelas": "tidak jelas", "nggak": "tidak",
    "klo": "kalau", "klw": "kalau", "klu": "kalau", "kok": "mengapa", "jgn": "jangan", "dr": "dari", 
    "utk": "untuk", "dgn": "dengan", "sndri": "sendiri", "aja": "saja", "doang": "saja", 
    "buat": "untuk", "bikin": "membuat", "udah": "sudah", "udh": "sudah", "uda": "sudah", 
    "jeleeeek": "jelek", "jelek": "buruk", "buruk": "buruk", "mau": "ingin", "tolong": "mohon", 
    "ya": "iya", "yah": "ya", "gimna": "bagaimana", "gini": "begini", "pas": "saat", 
    "blok": "goblok", "eror": "error", "jeblug": "error", "pake": "pakai", "nggo": "pakai",
    "org": "orang", "mntp": "mantap", "tdk": "tidak", "good": "baik", "good game": "permainan baik",
    "unuius wenda": "kerja bagus", "sangat luar biasa": "sangat bagus sekali",
    "rasah": "tidak perlu", "ungah": "unggah", "steady": "stabil", "testo": "percobaan mantab",
    "kntol": "jelek", "tolol": "tidak berguna", "kureng": "kurang", "binhung": "bingung", "bingung": "bingung",
    "fast": "cepat", "bangetttt": "banget", "bgt": "banget", "rame": "ramai", 
    "sy": "saya", "gw": "saya", "x": "nya", "matuk": "masuk", "mencla mencle": "tidak konsisten", 
    "bobrokny": "rusak", "bapuk": "buruk", "nek": "kalau", "ra": "tidak", "oke": "oke", 
    "bener": "benar", "cape": "capek", "lgi": "lagi", "lg": "lagi", "d": "di", "trus": "terus", "trs": "terus", 
    "abal": "tidak profesional", "abal2": "tidak profesional", "prah": "parah", "sumpeh": "sungguh", 
    "sumpah": "sungguh", "bkn": "bukan", "sbg": "sebagai", "hrs": "harus", "nyusahin": "menyusahkan",
    "dibenerein": "dibenarkan", "dihapus": "dihapuskan", "blom": "belum", "blm": "belum", 
    "nryimpeti": "menyulitkan", "minim": "minimal", "difoto": "di foto", "padahal": "meskipun",
    "loading lamaaaaa": "loading lama", "jelas banget": "sangat jelas", "amburadul": "kacau", 
    "dimanual": "dilakukan manual", "gampang": "mudah", "susah": "sulit", "nyusahi": "menyusahkan", 
    "rausah": "tidak usah", "durung": "belum", "buosokkk": "busuk", "ky": "kayak", "donwlod": "download", 
    "romusa": "kerja paksa", "ampas": "tidak berguna", "cuman": "cuma", "gitu": "seperti itu", 
    "ny": "nya", "jd": "jadi", "dlu": "dulu", "tu": "itu", "tp": "tapi", "sok\"an": "sok-sokan",
    "mark up": "markup", "embel embel": "tambahan tidak perlu", "nyiptain": "menciptakan", 
    "masuk,,": "masuk", "suka": "sering", "digelembungkan": "dilebihkan", "paling parah": "sangat buruk", 
    "ohhh": "", "si": "", "sih": "", "mah": "", "lah": "", "samsek": "sama sekali", "ora mutu":" tidak bermutu",
    "nyuruh": "menyuruh", "nyeselin": "menyebalkan", "nyecan": "scan", "gausah":"jangan", 
    "malem": "malam", "siang": "siang", "masyaallah": "", "berkalikali": "berkali-kali", "buramlog": "buram", 
    "SILIEUR INI MAH": "", "ngrepoti": "merepotkan", "mgaririweh": "merepotkan", "ngien": "membuat", "digunakanhadeuhhhhhhhhhhhhhhhhhh": "digunakan",
    "asal jadi": "tidak dirancang dengan baik", "bohhh": "muak", "karepmu": "urusanmu", "quick": "cepat",
    "server down": "server mati", "lemot": "lambat", "benerin": "perbaiki", "ngulang": "mengulang", "buduk": "buruk",
    "dewek": "sendiri", "Rungkad": "gagal", "burukkk": "buruk", "berkalikali": "berkali-kali", "masyaallah": "", "buramlog": "buram",
    "lemoot": "lemot", "lemottt": "lemot", "budug": "buruk", "buduk": "buruk", "anjrit": "sensor", "burammm": "buram",
    "downlagi": "server down", "geger": "ribut", "mumet": "pusing", "rungkat": "gagal", "rungkad": "gagal", "ampasss": "tidak berguna",
    "dewekan": "sendiri", "nyusahke": "menyusahkan", "ngelagg": "ngelag", "ngelegggggggggg": "ngelag", "burukkkkkkkkkk": "buruk",
    "gagalmaning": "gagal lagi", "Burruuuuuuukkkk":"buruk", "bagua": "bagus", "burul": "buruk", "pusingggg": "pusing", "capek2": "lelah",
    "uploadnya": "upload", "sdkt": "sedikit", "disaat": "saat", "terimakasih": "terima kasih", "udahh": "sudah", "besoknya": "besok",
    "rebet": "ribet", "palak": "memalak", "ngijo": "hijau", "nyo": "nya", "dupdate": "diupdate",
    "mhon": "mohon", "dperbaiki": "diperbaiki", "sat": "saat", "settt": "set", "lgsg": "langsung", "reset": "reset", "kontek": "kontak",
    "adeuh": "aduh", "teruss": "terus", "terussss": "terus", "payahhh": "payah", "mk": "makasih", "dibales": "dibalas",
    "kak": "kakak", "via": "melalui", "bs": "bisa", "sm": "sama", "lho": "loh", "gaes": "guys", "hbs": "habis",
    "sruh": "suruh", "sblmnya": "sebelumnya", "tambahin": "tambah", "mdh2n": "mudah-mudahan", "kmren": "kemarin", "tengkyu": "terima kasih",
    "hpnya": "hp", "cacattt": "cacat", "maap": "maaf", "numpang": "numpang", "bagussws": "bagus", "disuru": "disuruh",
    "anehh": "aneh", "aslii": "asli",  "burukdi": "buruk di", "submiteror": "submit error", "parahhhh": "parah",
    "semacamnya": "sejenisnya", "duit": "uang", "negera": "negara","rapih": "rapi", "bilang": "mengatakan", "standar": "standar", "bermanfaatsemoga": "bermanfaat semoga"
}

def bersihkan_teks(teks):
    
    # Normalisasi
    teks = str(teks).lower()
    for kata, pengganti in norm.items():
        teks = re.sub(r'\b' + re.escape(kata) + r'\b', pengganti, teks)
    # Lowercase
    #teks = teks.lower()
    # hapus URL
    teks = re.sub(r'https?://\S+|www\.\S+', '', teks)  
    # Hapus karakter non-alfabet
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    # Hapus spasi berlebih
    teks = re.sub(r'\s+', ' ', teks).strip()
    # Tokenisasi dan hapus stopwords
    tokens = teks.split()
    tokens_bersih = [t for t in tokens if t not in stopwords_indonesia]
    return ' '.join(tokens_bersih)

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

# --- Preprocessing ---
@app.route('/preprocessing-labeling', methods=['GET', 'POST'])
@login_required
def preprocessing():
    hasil = []

    # Jika user menekan tombol "Lihat Data"
    if request.method == 'GET' and request.args.get('action') == 'lihat':
        hasil_data = PreprosesData.query.order_by(PreprosesData.timestamp.desc()).all()
        if not hasil_data:
            flash("Belum ada data hasil preprocessing yang bisa ditampilkan.", 'warning')
            return redirect(url_for('preprocessing'))

        for item in hasil_data:
            hasil.append({
                'original': item.original,
                'cleaned': item.cleaned,
                'label': item.label,
                'timestamp': item.timestamp.strftime("%d-%m-%Y %H:%M") if item.timestamp else "-"
            })
        flash(f"Menampilkan {len(hasil)} data hasil preprocessing.", 'info')

    # Jika user menekan tombol "Proses"
    elif request.method == 'POST':
        ulasan_list = Ulasan.query.order_by(Ulasan.timestamp.desc()).all()

        if not ulasan_list:
            flash("Belum ada data untuk diproses.", 'warning')
            return redirect(request.url)

        data_baru_ditemukan = False

        for ulasan in ulasan_list:
            # Cek apakah ulasan sudah diproses
            existing = PreprosesData.query.filter_by(ulasan_id=ulasan.id).first()
            if existing:
                continue

            data_baru_ditemukan = True
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

        if not data_baru_ditemukan:
            flash("Semua data ulasan sudah pernah diproses. Tidak ada data baru.", 'info')
            return redirect(request.url)

        db.session.commit()
        flash(f"{len(hasil)} data ulasan berhasil diproses.", 'success')
        
    return render_template(
    'preprocessing_labeling.html',
    title='Preprocessing dan Labeling Data',
    hasil=hasil,
    filename=filename if 'filename' in locals() else None
)
# Download preprocessed
@app.route('/download-preprocessed')
@login_required
def download_preprocessed():
    from io import StringIO
    import csv

    data = PreprosesData.query.order_by(PreprosesData.timestamp.desc()).all()
    
    if not data:
        flash("Tidak ada data hasil preprocessing untuk diunduh.", 'warning')
        return redirect(url_for('preprocessing'))

    # Simpan ke memori (bukan file fisik di server)
    si = StringIO()
    writer = csv.writer(si)
    
    # Header
    writer.writerow(['original', 'cleaned', 'label', 'timestamp'])

    # Isi data
    for item in data:
        writer.writerow([
            item.original,
            item.cleaned,
            item.label,
            item.timestamp.strftime('%Y-%m-%d %H:%M:%S') if item.timestamp else '-'
        ])

    # Kirim file CSV sebagai response untuk diunduh
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment;filename=preprocessed_and_labeling_data.csv"
        }
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
            flash("Belum ada data untuk diproses.")
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

# Jika data kosong, tampilkan notifikasi
    if not data or all(not row[0] for row in data):
        flash("Tidak ada data yang tersedia untuk Word Cloud.", "warning")
        return redirect(url_for('visualisasi'))
    # Buat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
# Gabungkan semua teks
    all_text = ' '.join([row[0] for row in data if row[0]])
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

# --- Periode --- 
@app.route('/periode')
@login_required
def periode():
    # Ambil seluruh timestamp dari tabel Ulasan
    rows = db.session.query(Ulasan.timestamp).filter(Ulasan.timestamp != None).all()
    df = pd.DataFrame(rows, columns=['timestamp'])
    if df.empty:
        flash("Belum ada data ulasan untuk divisualisasikan.", "warning")
        return redirect(url_for('visualisasi'))

    # Pastikan tipe datetime dan buat kolom 'tanggal'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['tanggal'] = df['timestamp'].dt.date

    # Hitung jumlah ulasan per hari
    per_hari = df.groupby('tanggal').size().reset_index(name='jumlah')

    # Buat grafik line chart
    plt.figure(figsize=(10, 5))
    plt.plot(per_hari['tanggal'], per_hari['jumlah'], marker='o')
    plt.title('Jumlah Ulasan Per Hari')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Konversi grafik ke gambar base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # Kirim ke template
    return render_template('periode.html', title="Periode", plot_url=img_b64)



# =========================
# === END OF ROUTES =======
# =========================

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)