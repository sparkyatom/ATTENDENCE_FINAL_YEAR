from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import base64
from datetime import datetime

app = Flask(__name__)

# Ensure the upload directory exists
UPLOAD_FOLDER = 'static/input_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the main dashboard page."""
    # Get all images from the input_images folder
    images = []
    if os.path.exists(UPLOAD_FOLDER):
        images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    
    return render_template('index.html', images=images)

@app.route('/capture', methods=['POST'])
def capture():
    """Handle the image capture from webcam."""
    try:
        # Get the image data from the request
        image_data = request.json.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data received'})
        
        # Remove the data URL prefix to get the base64 data
        image_data = image_data.split(',')[1]
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'filepath': f'/static/input_images/{filename}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    """Delete an image from the input_images folder."""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)