// static/js/webcam.js
document.addEventListener('DOMContentLoaded', function() {
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const captureButton = document.getElementById('capture-btn');
    const switchCameraButton = document.getElementById('switch-camera-btn');
    const capturedImage = document.getElementById('captured-image');
    const noImageMsg = document.getElementById('no-image-msg');
    const imageGallery = document.getElementById('image-gallery');
    
    let currentStream;
    let facingMode = 'user'; // Start with front camera

    // Start the webcam
    async function startWebcam() {
        try {
            const constraints = {
                video: {
                    facingMode: facingMode
                }
            };

            // Stop any existing stream
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
            }

            // Start a new stream
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            webcamElement.srcObject = stream;
            currentStream = stream;
            
            // Enable capture button
            captureButton.disabled = false;
        } catch (error) {
            console.error('Error starting webcam:', error);
            alert('Unable to access the webcam. Please make sure you have a webcam connected and have granted permission to use it.');
            captureButton.disabled = true;
        }
    }

    // Switch between front and back cameras
    switchCameraButton.addEventListener('click', function() {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        startWebcam();
    });

    // Capture image from webcam
    captureButton.addEventListener('click', async function() {
        // Set canvas dimensions to match video dimensions
        const width = webcamElement.videoWidth;
        const height = webcamElement.videoHeight;
        canvasElement.width = width;
        canvasElement.height = height;
        
        // Draw current webcam frame to canvas
        const context = canvasElement.getContext('2d');
        context.drawImage(webcamElement, 0, 0, width, height);
        
        // Get image data from canvas
        const imageData = canvasElement.toDataURL('image/png');
        
        // Display captured image
        capturedImage.src = imageData;
        capturedImage.classList.remove('d-none');
        noImageMsg.classList.add('d-none');
        
        // Send image to server
        try {
            const response = await fetch('/capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Add the new image to the gallery
                addImageToGallery(data.filename, data.filepath);
            } else {
                console.error('Error saving image:', data.error);
                alert('Failed to save the image. Please try again.');
            }
        } catch (error) {
            console.error('Error sending image to server:', error);
            alert('Failed to send the image to the server. Please check your connection and try again.');
        }
    });

    // Add new image to gallery
    function addImageToGallery(filename, filepath) {
        // Check if "No images available" message is present and remove it
        const noImagesMsg = document.querySelector('#image-gallery .col-12');
        if (noImagesMsg && noImagesMsg.textContent.trim() === 'No images available') {
            noImagesMsg.remove();
        }
        
        // Create new gallery item
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-6 mb-3';
        colDiv.innerHTML = `
            <div class="card">
                <img src="${filepath}" class="card-img-top" alt="Captured Image">
                <div class="card-body p-2 text-center">
                    <p class="card-text small">${filename}</p>
                    <form method="post" action="/delete/${filename}">
                        <button type="submit" class="btn btn-danger btn-sm">Delete</button>
                    </form>
                </div>
            </div>
        `;
        
        // Add to the beginning of the gallery
        if (imageGallery.firstChild) {
            imageGallery.insertBefore(colDiv, imageGallery.firstChild);
        } else {
            imageGallery.appendChild(colDiv);
        }
    }

    // Start webcam when page loads
    startWebcam();
});