import os
import cloudinary
import cloudinary.uploader
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['SECRET_KEY'] = 'your_secret_key'

# Cloudinary Configuration
cloudinary.config(
    cloud_name='dri1k0jz1',
    api_key='218999275722715',
    api_secret='0bCEbzkVso5j04WwUBXGdsakai4'
)

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Get form data
        roll = request.form.get('roll')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        institute = request.form.get('institute')
        stream = request.form.get('stream')
        year = request.form.get('year')
        email = request.form.get('email')
        phone = request.form.get('phone')

        # Handle selfie upload to Cloudinary
        selfie_url = None
        if 'selfie' in request.files:
            selfie = request.files['selfie']
            if selfie.filename != '':
                filename = secure_filename(f"{roll}_selfie.jpg")
                upload_result = cloudinary.uploader.upload(selfie, folder='uploads/images/', public_id=f"{roll}_selfie")
                selfie_url = upload_result.get('secure_url')

        # Handle video upload to Cloudinary
        video_url = None
        if 'video' in request.files:
            video = request.files['video']
            if video.filename != '':
                video_filename = secure_filename(f"{roll}_video.webm")
                upload_result = cloudinary.uploader.upload(video, folder='uploads/videos/', public_id=f"{roll}_video", resource_type='video')
                video_url = upload_result.get('secure_url')

        # Here you would typically save form data + URLs to a database

        return jsonify({
            'success': True,
            'message': 'Form submitted successfully!',
            'selfie_url': selfie_url,
            'video_url': video_url
        })

    return jsonify({'success': False, 'message': 'Something went wrong!'})

@app.route('/save-video', methods=['POST'])
def save_video():
    if 'video' in request.files:
        video = request.files['video']
        roll = request.form.get('roll', 'unknown')
        if video.filename != '':
            video_filename = secure_filename(f"{roll}_video.webm")
            upload_result = cloudinary.uploader.upload(video, folder='uploads/videos/', public_id=f"{roll}_video", resource_type='video')
            return jsonify({'success': True, 'video_url': upload_result.get('secure_url')})

    return jsonify({'success': False, 'message': 'Video upload failed!'})

if __name__ == '__main__':
    app.run(debug=True)
