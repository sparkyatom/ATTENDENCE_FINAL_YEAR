document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const form = document.getElementById('studentForm');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const retakeBtn = document.getElementById('retakeBtn');
    const webcamVideo = document.getElementById('webcam');
    const recordedVideo = document.getElementById('recorded-video');
    const cameraContainer = document.getElementById('camera-container');
    const recordedVideoContainer = document.getElementById('recorded-video-container');
    const selfieInput = document.getElementById('selfie');
    const selfiePreview = document.getElementById('selfiePreview');
    const rollInput = document.getElementById('roll');
    
    // Global variables
    let mediaRecorder;
    let recordedBlobs = [];
    let stream;
    let videoFile = null;
    
    // Initialize webcam
    async function initializeWebcam() {
        try {
            const constraints = {
                audio: true,
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            webcamVideo.srcObject = stream;
            
            startBtn.disabled = false;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            alert('Unable to access your webcam. Please check your permissions and try again.');
        }
    }
    
    // Start recording
    function startRecording() {
        recordedBlobs = [];
        
        try {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        } catch (error) {
            console.error('MediaRecorder error:', error);
            alert('Your browser does not support recording. Please try another browser.');
            return;
        }
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedBlobs.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            const videoBlob = new Blob(recordedBlobs, { type: 'video/webm' });
            recordedVideo.src = URL.createObjectURL(videoBlob);
            
            // Create a File object from the Blob
            const fileName = `video_${new Date().getTime()}.webm`;
            videoFile = new File([videoBlob], fileName, { type: 'video/webm' });
        };
        
        // Start recording
        mediaRecorder.start(100);
        
        // UI changes
        startBtn.classList.add('d-none');
        stopBtn.classList.remove('d-none');
        
        // Auto-stop after 30 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            }
        }, 30000);
    }
    
    // Stop recording
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            
            // UI changes
            stopBtn.classList.add('d-none');
            retakeBtn.classList.remove('d-none');
            cameraContainer.classList.add('d-none');
            recordedVideoContainer.classList.remove('d-none');
        }
    }
    
    // Retake video
    function retakeVideo() {
        // UI changes
        cameraContainer.classList.remove('d-none');
        recordedVideoContainer.classList.add('d-none');
        startBtn.classList.remove('d-none');
        retakeBtn.classList.add('d-none');
        
        // Clear recorded video
        recordedVideo.src = '';
        videoFile = null;
    }
    
    // Display selfie preview
    function updateSelfiePreview() {
        if (selfieInput.files && selfieInput.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                selfiePreview.src = e.target.result;
                selfiePreview.classList.remove('d-none');
            };
            reader.readAsDataURL(selfieInput.files[0]);
        } else {
            selfiePreview.classList.add('d-none');
        }
    }
    
    // Submit form
    async function submitForm(event) {
        event.preventDefault();
        
        // Form validation
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        
        // Check if video is recorded
        if (!videoFile) {
            alert('Please record a video before submitting.');
            return;
        }
        
        // Prepare form data
        const formData = new FormData(form);
        
        // Add video file
        formData.append('video', videoFile);
        
        try {
            // Upload data
            const response = await fetch('/submit', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Show success modal
                const successModal = new bootstrap.Modal(document.getElementById('successModal'));
                successModal.show();
                
                // Reset form
                form.reset();
                selfiePreview.classList.add('d-none');
                retakeVideo();
            } else {
                alert('Error submitting form: ' + result.message);
            }
        } catch (error) {
            console.error('Error submitting form:', error);
            alert('An error occurred while submitting the form. Please try again.');
        }
    }
    
    // Event listeners
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    retakeBtn.addEventListener('click', retakeVideo);
    selfieInput.addEventListener('change', updateSelfiePreview);
    form.addEventListener('submit', submitForm);
    
    // Initialize webcam on page load
    initializeWebcam();
});
