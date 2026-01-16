# SIREKAP Sentiment Analysis System 🗳️

Sistem Analisis Sentimen berbasis Web yang dikembangkan untuk mengevaluasi kinerja aplikasi SIREKAP Mobile pada Pilkada Kabupaten Tulungagung 2024. Proyek ini menggabungkan pengembangan web dengan Machine Learning untuk memberikan wawasan berbasis data.

## 🚀 Key Features
- **Automated Scraping:** Pengambilan ulasan secara real-time dari Google Play Store menggunakan `google-play-scraper`.
- **Machine Learning Core:** Menggunakan algoritma **Naïve Bayes** dengan ekstraksi fitur **TF-IDF**.
- [cite_start]**High Performance:** Mencapai tingkat akurasi sebesar **83.5%** dalam klasifikasi sentimen[cite: 17].
- **Interactive Dashboard:** Visualisasi data menggunakan WordCloud dan grafik distribusi sentimen untuk analisis cepat.
- **REST API Support:** Menyediakan endpoint API untuk integrasi dengan platform pihak ketiga.

## 🛠️ Tech Stack
- **Backend:** Python (Flask Framework)
- **API:** REST API dengan format JSON
- **Database:** MySQL dengan SQLAlchemy ORM
- **ML Libraries:** Scikit-learn, NLTK, Sastrawi (Indonesian Stemmer)
- **Frontend:** HTML5, CSS3, JavaScript (Bootstrap-based)
- **Security:** Flask-Login & Werkzeug Password Hashing

## 🔌 API Documentation
Aplikasi ini kini mendukung layanan API untuk prediksi sentimen tanpa melalui antarmuka web.

**Endpoint:** `POST /api/predict`
**Contoh Request (JSON):**
```json
{
  "ulasan": "Aplikasi SIREKAP sangat membantu!"
}

Tentu, ini adalah versi README.md yang sudah diperbarui dan dipolitisasi. Saya telah menambahkan bagian khusus untuk REST API yang baru saja kamu buat dan berhasil kamu tes di Postman.

Silakan salin kode di bawah ini ke file README.md di repositori GitHub-mu:

Markdown

# SIREKAP Sentiment Analysis System 🗳️

Sistem Analisis Sentimen berbasis Web yang dikembangkan untuk mengevaluasi kinerja aplikasi SIREKAP Mobile pada Pilkada Kabupaten Tulungagung 2024. Proyek ini menggabungkan pengembangan web dengan Machine Learning untuk memberikan wawasan berbasis data.

## 🚀 Key Features
- **Automated Scraping:** Pengambilan ulasan secara real-time dari Google Play Store menggunakan `google-play-scraper`.
- **Machine Learning Core:** Menggunakan algoritma **Naïve Bayes** dengan ekstraksi fitur **TF-IDF**.
- [cite_start]**High Performance:** Mencapai tingkat akurasi sebesar **83.5%** dalam klasifikasi sentimen[cite: 17].
- **Interactive Dashboard:** Visualisasi data menggunakan WordCloud dan grafik distribusi sentimen untuk analisis cepat.
- **REST API Support:** Menyediakan endpoint API untuk integrasi dengan platform pihak ketiga.

## 🛠️ Tech Stack
- **Backend:** Python (Flask Framework)
- **API:** REST API dengan format JSON
- **Database:** MySQL dengan SQLAlchemy ORM
- **ML Libraries:** Scikit-learn, NLTK, Sastrawi (Indonesian Stemmer)
- **Frontend:** HTML5, CSS3, JavaScript (Bootstrap-based)
- **Security:** Flask-Login & Werkzeug Password Hashing

## 🔌 API Documentation
Aplikasi ini kini mendukung layanan API untuk prediksi sentimen tanpa melalui antarmuka web.

**Endpoint:** `POST /api/predict`
**Contoh Request (JSON):**
```json
{
  "ulasan": "Aplikasi SIREKAP sangat membantu!"
}

## Contoh Response:
{
  "status": "success",
  "input": "Aplikasi SIREKAP sangat membantu!",
  "sentiment": "Positif"
}
