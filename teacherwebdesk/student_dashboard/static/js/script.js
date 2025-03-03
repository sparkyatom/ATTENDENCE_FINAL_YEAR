// Global variables
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

// DOM elements
const webcamVideo = document.getElementById('webcam');
const recordingPreview = document.getElementById('recordingPreview');
const startButton = document.getElementById('startButton');
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const uploadButton = document.getElementById('uploadButton');
const recordingIndicator = document.getElementById('recording-indicator');
const recordingStatus = document.getElementById('recordingStatus');
const uploadStatus = document.getElementById('uploadStatus');
const errorStatus = document.getElementById('errorStatus');
const previousRecordings = document.getElementById('previousRecordings');
const noRecordings = document.getElementById('noRecordings');

// Start camera
startButton.addEventListener('click', async () => {
    try {
        // Request camera access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
        
        // Display camera feed
        webcamVideo.srcObject = mediaStream;
        
        // Enable record button
        recordButton.disabled = false;
        startButton.disabled = true;
        
        hideAllAlerts();
    } catch (error) {
        showError('Camera access denied or not available: ' + error.message);
    }
});

// Start recording
recordButton.addEventListener('click', () => {
    // Create MediaRecorder
    recordedChunks = [];
    const options = { mimeType: 'video/webm' };
    
    try {
        mediaRecorder = new MediaRecorder(mediaStream, options);
    } catch (error) {
        showError('MediaRecorder error: ' + error.message);
        return;
    }
    
    // Handle data available event
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    
    // Handle recording stop
    mediaRecorder.onstop = () => {
        // Create blob from recorded chunks
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        
        // Display recording preview
        recordingPreview.src = url;
        recordingPreview.play();
        
        // Show upload button
        uploadButton.classList.remove('d-none');
        
        // Update UI
        isRecording = false;
        recordingIndicator.classList.add('d-none');
        recordingStatus.classList.add('d-none');
        recordButton.classList.remove('d-none');
        stopButton.classList.add('d-none');
    };
    
    // Start recording
    mediaRecorder.start();
    isRecording = true;
    
    // Update UI
    recordButton.classList.add('d-none');
    stopButton.classList.remove('d-none');
    recordingIndicator.classList.remove('d-none');
    recordingStatus.classList.remove('d-none');
    uploadButton.classList.add('d-none');
    hideAllAlerts();
});

// Stop recording
stopButton.addEventListener('click', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
    }
});

// Upload recording
uploadButton.addEventListener('click', () => {
    if (recordedChunks.length === 0) {
        showError('No recording to upload');
        return;
    }
    
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('video', blob, 'recording.webm');
    
    // Show loading state
    uploadButton.disabled = true;
    uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
    
    // Upload to server
    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Show success message
        uploadStatus.classList.remove('d-none');
        setTimeout(() => {
            uploadStatus.classList.add('d-none');
        }, 3000);
        
        // Reset upload button
        uploadButton.disabled = false;
        uploadButton.innerHTML = '<i class="bi bi-cloud-upload"></i> Upload Recording';
        uploadButton.classList.add('d-none');
        
        // Load all videos from server
        loadVideos();
    })
    .catch(error => {
        showError('Upload failed: ' + error.message);
        uploadButton.disabled = false;
        uploadButton.innerHTML = '<i class="bi bi-cloud-upload"></i> Upload Recording';
    });
});

// Load videos from server
function loadVideos() {
    fetch('/videos')
    .then(response => response.json())
    .then(videos => {
        if (videos.length > 0) {
            // Hide the "no recordings" message
            noRecordings.classList.add('d-none');
            
            // Clear current videos
            previousRecordings.innerHTML = '';
            
            // Create video elements for each recording
            videos.forEach(video => {
                const videoElement = document.createElement('div');
                videoElement.className = 'col-md-4 col-sm-6 video-item';
                
                videoElement.innerHTML = `
                    <div class="card h-100 shadow-sm">
                        <video class="video-thumbnail" src="${video.path}" controls></video>
                        <div class="card-body">
                            <h5 class="card-title">Take ${video.take_number}</h5>
                            <p class="card-text small text-muted">${video.timestamp}</p>
                        </div>
                    </div>
                `;
                
                previousRecordings.appendChild(videoElement);
            });
        } else {
            // Show the "no recordings" message
            noRecordings.classList.remove('d-none');
        }
    })
    .catch(error => {
        showError('Failed to load videos: ' + error.message);
    });
}

// Helper functions
function hideAllAlerts() {
    recordingStatus.classList.add('d-none');
    uploadStatus.classList.add('d-none');
    errorStatus.classList.add('d-none');
}

function showError(message) {
    errorStatus.textContent = message;
    errorStatus.classList.remove('d-none');
    setTimeout(() => {
        errorStatus.classList.add('d-none');
    }, 5000);
}

// Load existing videos on page load
document.addEventListener('DOMContentLoaded', loadVideos);

// Clean up when leaving the page
window.addEventListener('beforeunload', () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
});