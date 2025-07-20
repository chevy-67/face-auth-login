#from flask import Flask, request, render_template, jsonify
from flask import *
import cv2
import face_recognition as fr
import numpy as np
import sqlite3
import base64
import joblib as jb
from tensorflow.keras.models import load_model

app = Flask(__name__)

DATABASE = 'users.db'

def init_db():
    con = sqlite3.connect(DATABASE)
    c = con.cursor()
    query = 'CREATE TABLE IF NOT EXISTS users (username TEXT UNIQUE, password TEXT)'
    c.execute(query)
    con.commit()
    con.close()

init_db()

def save_face_encoding(username):
    image_path = f'captured/{username}.jpg'
    image = cv2.imread(image_path)
    image_rgb  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    face_encodings = fr.face_encodings(image_rgb)
    if not face_encodings:
        return render_template('signup.html',msg='No face detected in the photo')
    encoding_path = f'encodings/{username}.npy'
    np.save(encoding_path, face_encodings[0])
    print(f"Face encoding saved at: {encoding_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def home1():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        image_data = request.form['captured_image']
        image_path = f'captured/{username}.jpg'
        password = request.form['pass']
        cap_image = request.files.get('image-file')

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users")
        users = cursor.fetchall()
        conn.close()

        if username in users:
            return render_template('signup.html',msg='Username already exists')

        if(cap_image):
            cap_image.save(image_path)
        else:
            try:
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_data.split(",")[1]))
            except Exception as e:
                "Failed to save webcam image: {e}"
        try:
            save_face_encoding(username)
        except ValueError as e:
            return f"Error: {e}"
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                           (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template('signup.html',msg="Username already exists")
        finally:
            conn.close()

        return render_template('index.html',msg="Signup successful")
    return render_template('signup.html',msg='')



@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        image = request.form['image']
        password = request.form['pass']
        temp_image_path = f'captured/temp.jpg'

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT username,password from users WHERE username = ?',(username,))
        user = cursor.fetchone()
        conn.close()

        with open(temp_image_path,'wb') as img:
            img.write(base64.b64decode(image.split(',')[1]))
        
        temp_image = cv2.imread(temp_image_path)
        temp_image_rgb = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        face_encodings = fr.face_encodings(temp_image_rgb)

        if not face_encodings:
            msg = "No face detected."
            return render_template('login.html',msg=msg)
        temp_encoding = face_encodings[0]

        model = load_model('models/model_v1.6.h5')

        def predict(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img,(224,224))/255.0
            img = np.expand_dims(img,axis=0)
            pred = model.predict(img)

            return pred[0]>0.95

        if not user:
            msg="User not exists!"
            return render_template('login.html',msg=msg)

        username_in_db,password_in_db = user

        if password!=password_in_db:
            msg="Wrong password"
            return render_template('login.html',msg=msg)
        
        encoding_path = f'encodings/{username_in_db}.npy'
        saved_encoding = np.load(encoding_path)
        matches = fr.compare_faces([saved_encoding], temp_encoding)
        face_dist = fr.face_distance([saved_encoding], temp_encoding)[0]

        if matches[0] and face_dist < 0.4:
                    if predict(temp_image_path):
                        return render_template('home.html',username=username_in_db)
                    else:
                        msg="Face spoof detected"
                        return render_template('login.html',msg=msg)

        msg = "Face Mismatch!"
        return render_template('login.html',msg=msg)
    return render_template('login.html',msg=msg)


if __name__ == '__main__':
    app.run(debug=True)
