from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
from face_embedding_extraction import get_embedding
import json
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize Database
db = SQLAlchemy(app)

# Database Model
class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), unique=True, nullable=False)
    face_embeddings = db.Column(db.Text, nullable=True)  # Added face_embeddings column

# Create the Database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')  # Assuming you have a results.html template

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        user_name = request.form.get('user_name')
        files = request.files.getlist('images')

        if not user_name or not files:
            flash("Please fill all fields and select images", 'warning')
            return redirect(url_for('index'))

        embeddings = []
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                image = cv2.imread(file_path)
                embedding = get_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
                if os.path.exists(file_path):
                    os.remove(file_path)

        if embeddings:
            avg_embeddings = np.mean(embeddings, axis=0)
            avg_embeddings_json = json.dumps(avg_embeddings.tolist())

            existing_entry = ImageUpload.query.filter_by(user_name=user_name).first()
            if existing_entry:
                existing_entry.face_embeddings = avg_embeddings_json
            else:
                new_entry = ImageUpload(user_name=user_name, face_embeddings=avg_embeddings_json)
                db.session.add(new_entry)

        if embeddings:
            avg_embeddings = np.mean(embeddings, axis=0)
            avg_embeddings_json = json.dumps(avg_embeddings.tolist())

            existing_entry = ImageUpload.query.filter_by(user_name=user_name).first()
            if existing_entry:
                existing_entry.face_embeddings = avg_embeddings_json
            else:
                new_entry = ImageUpload(user_name=user_name, face_embeddings=avg_embeddings_json)
                db.session.add(new_entry)

        db.session.commit()
        return redirect(url_for('index'))  # Ensure a valid response is returned
        flash("Images processed successfully!", 'success')
        return redirect(url_for('index'))  # Ensure a valid response is returned
    else:
        flash("No valid images were processed.", 'warning')
        return redirect(url_for('index'))  # Ensure a valid response is returned

if __name__ == '__main__':
    print("Available routes:")
    print(" - /")
    print(" - /upload")
    print(" - /results")
    app.run(debug=True)
