# config.py


DB_USER = 'root'
DB_PASSWORD = 'anjay123'
DB_HOST = 'localhost'
DB_NAME = 'sirekap_sentimen'

SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://root:anjay123@localhost/Sentimen_Sirekap'
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = os.environ.get('SECRET_KEY', 'rahasia-aman-123')