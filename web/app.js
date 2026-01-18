// Configuration
const API_URL = 'http://localhost:5000';
const CAPTURE_FPS = 3; // Reduced FPS for more stable predictions
const MAX_HISTORY = 10; // Maximum number of predictions to keep in history
const MIN_CONFIDENCE_FOR_HISTORY = 0.80; // Increased threshold for history

// Global state
let videoStream = null;
let captureInterval = null;
let isCapturing = false;
let fpsInterval = null;
let frameCount = 0;
let lastPrediction = null;

// DOM Elements
const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const videoOverlay = document.getElementById('videoOverlay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const connectionStatus = document.getElementById('connectionStatus');
const fpsCounter = document.getElementById('fpsCounter');
const predictionText = document.getElementById('predictionText');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const topPredictionsList = document.getElementById('topPredictionsList');
const historyList = document.getElementById('historyList');

// Prediction history
let predictionHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIConnection();
});

// Event Listeners
function setupEventListeners() {
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    clearBtn.addEventListener('click', clearHistory);
}

// Check API Connection
async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            updateConnectionStatus(true);
        } else {
            updateConnectionStatus(false);
        }
    } catch (error) {
        console.error('API connection error:', error);
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatus.textContent = 'Connected';
        connectionStatus.classList.add('connected');
        connectionStatus.classList.remove('disconnected');
    } else {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.classList.add('disconnected');
        connectionStatus.classList.remove('connected');
    }
}

// Camera Functions
async function startCamera() {
    try {
        // Request camera access
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });

        // Set video source
        videoElement.srcObject = videoStream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });

        // Hide overlay
        videoOverlay.classList.add('hidden');

        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isCapturing = true;

        // Start capturing frames
        startFrameCapture();
        startFPSCounter();

        console.log('Camera started successfully');
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Failed to access camera. Please ensure you have granted camera permissions.');
    }
}

function stopCamera() {
    // Stop video stream
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    // Stop capture interval
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    // Stop FPS counter
    if (fpsInterval) {
        clearInterval(fpsInterval);
        fpsInterval = null;
    }

    // Reset video element
    videoElement.srcObject = null;

    // Show overlay
    videoOverlay.classList.remove('hidden');

    // Update UI
    startBtn.disabled = false;
    stopBtn.disabled = true;
    isCapturing = false;
    fpsCounter.textContent = '0';

    console.log('Camera stopped');
}

// Frame Capture
function startFrameCapture() {
    const captureIntervalMs = 1000 / CAPTURE_FPS;

    captureInterval = setInterval(async () => {
        if (isCapturing && videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
            await captureAndPredict();
        }
    }, captureIntervalMs);
}

async function captureAndPredict() {
    try {
        // Set canvas dimensions to match video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

        // Draw current frame to canvas
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);

        // Convert canvas to base64 with higher quality
        const imageData = canvasElement.toDataURL('image/jpeg', 0.95);

        // Send to API for prediction
        const response = await fetch(`${API_URL}/predict-frame`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                updatePredictionDisplay(data);
                frameCount++;
            }
        } else {
            console.error('Prediction request failed:', response.status);
            updateConnectionStatus(false);
        }
    } catch (error) {
        console.error('Error during prediction:', error);
        updateConnectionStatus(false);
    }
}

// Update UI with Prediction
function updatePredictionDisplay(data) {
    const { prediction, confidence, top_3 } = data;

    // Update main prediction
    predictionText.textContent = prediction;

    // Update confidence bar
    const confidencePercent = (confidence * 100).toFixed(1);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    // Update top 3 predictions
    if (top_3 && top_3.length > 0) {
        const items = topPredictionsList.querySelectorAll('.top-prediction-item');
        top_3.forEach((pred, index) => {
            if (items[index]) {
                const classNameEl = items[index].querySelector('.class-name');
                const confidenceEl = items[index].querySelector('.confidence');
                classNameEl.textContent = pred.class;
                confidenceEl.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
            }
        });
    }

    // Add to history (only if prediction changed or confidence is high)
    if (shouldAddToHistory(prediction, confidence)) {
        addToHistory(prediction, confidence);
    }

    lastPrediction = { prediction, confidence };
}

function shouldAddToHistory(prediction, confidence) {
    // Add if confidence is above threshold and prediction changed
    if (confidence < MIN_CONFIDENCE_FOR_HISTORY) return false;
    if (!lastPrediction) return true;
    return lastPrediction.prediction !== prediction;
}

function addToHistory(prediction, confidence) {
    const timestamp = new Date().toLocaleTimeString();

    predictionHistory.unshift({
        prediction,
        confidence,
        timestamp
    });

    // Limit history size
    if (predictionHistory.length > MAX_HISTORY) {
        predictionHistory.pop();
    }

    renderHistory();
}

function renderHistory() {
    if (predictionHistory.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No predictions yet</div>';
        return;
    }

    historyList.innerHTML = predictionHistory.map(item => `
        <div class="history-item">
            <span class="history-class">${item.prediction}</span>
            <span class="history-confidence">${(item.confidence * 100).toFixed(1)}%</span>
            <span class="history-time">${item.timestamp}</span>
        </div>
    `).join('');
}

function clearHistory() {
    predictionHistory = [];
    renderHistory();
    console.log('History cleared');
}

// FPS Counter
function startFPSCounter() {
    frameCount = 0;
    fpsInterval = setInterval(() => {
        fpsCounter.textContent = frameCount;
        frameCount = 0;
    }, 1000);
}

// Periodic API health check
setInterval(checkAPIConnection, 10000); // Check every 10 seconds
