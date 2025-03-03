from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store video information in memory (in a real app, use a database)
videos = []

@app.route('/')
def index():
    return render_template('index.html', videos=videos)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the file
    video_file.save(filepath)
    
    # Store video information
    video_info = {
        'id': str(uuid.uuid4()),
        'filename': filename,
        'path': f"/static/uploads/{filename}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'take_number': len(videos) + 1
    }
    videos.append(video_info)
    
    return jsonify(video_info), 200

@app.route('/videos', methods=['GET'])
def get_videos():
    return jsonify(videos)

if __name__ == '__main__':
    app.run(debug=True)