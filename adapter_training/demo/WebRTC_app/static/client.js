const micBtn = document.getElementById('mic-btn');
const sendBtn = document.getElementById('send-btn');
const textInput = document.getElementById('text-input');
const messagesContainer = document.getElementById('messages');
const statusIndicator = document.getElementById('connection-status');

let pc = null;
let dc = null;
let localStream = null;
let isRecording = false;

// Initialize WebRTC
async function startWebRTC() {
    const config = {
        sdpSemantics: 'unified-plan'
    };

    pc = new RTCPeerConnection(config);

    // Handle Data Channel
    dc = pc.createDataChannel('chat');
    setupDataChannel(dc);

    // Handle ICE candidates (not strictly needed for local dev but good practice)
    pc.addEventListener('iceconnectionstatechange', () => {
        if (pc.iceConnectionState === 'failed') {
            pc.close();
            statusIndicator.textContent = 'Connection Failed';
            statusIndicator.classList.remove('connected');
        }
    });

    return pc;
}

function setupDataChannel(channel) {
    channel.onopen = () => {
        statusIndicator.textContent = 'Connected';
        statusIndicator.classList.add('connected');
    };

    channel.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        if (data.type === 'response') {
            appendMessage('system', data.text);
        }
    };
}

async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch('/offer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
        })
    });

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
}

// UI Interactions
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
    // Auto-resize
    textInput.style.height = 'auto';
    textInput.style.height = textInput.scrollHeight + 'px';
});

sendBtn.addEventListener('click', sendMessage);

const uploadBtn = document.getElementById('upload-btn');
const audioInput = document.getElementById('audio-upload');

uploadBtn.addEventListener('click', () => {
    audioInput.click();
});

audioInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const prompt = textInput.value.trim();
    textInput.value = '';

    appendMessage('user', `Uploaded: ${file.name} ${prompt ? `\nPrompt: ${prompt}` : ''}`);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('sessionId', sessionId);
    if (prompt) {
        formData.append('text', prompt);
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            appendMessage('system', data.text);
        } else {
            appendMessage('system', 'Error uploading file.');
        }
    } catch (e) {
        console.error('Upload error:', e);
        appendMessage('system', 'Error uploading file.');
    }

    // Reset input
    audioInput.value = '';
});

micBtn.addEventListener('click', async () => {
    if (!isRecording) {
        await startRecording();
    } else {
        await stopRecording();
    }
});

async function startRecording() {
    try {
        // Always start fresh to ensure server/client sync
        if (pc) {
            pc.close();
            pc = null;
        }

        await startWebRTC();

        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            },
            video: false
        });

        localStream.getTracks().forEach(track => {
            pc.addTrack(track, localStream);
        });

        await negotiate();

        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'; // Stop icon

    } catch (e) {
        console.error('Error starting recording:', e);
        alert('Could not access microphone.');
    }
}

async function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>'; // Mic icon

    // Stop tracks
    if (localStream) {
        localStream.getTracks().forEach(track => {
            track.stop();
            // Remove from PC? Not strictly necessary if we just stop sending
        });
    }

    // Signal server to process
    // We send the text prompt if any, or just empty string
    const prompt = textInput.value.trim();
    textInput.value = '';

    if (dc && dc.readyState === 'open') {
        dc.send(JSON.stringify({
            type: 'stop',
            text: prompt
        }));

        if (prompt) {
            appendMessage('user', prompt + ' (Audio)');
        } else {
            appendMessage('user', '(Audio Message)');
        }
    }

    // Do NOT close the PC here, otherwise we won't receive the response!
    // We keep the connection open for the session.
}

function sendMessage() {
    const text = textInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    textInput.value = '';
    textInput.style.height = 'auto';

    // If not recording, this is a text-only message
    if (!isRecording) {
        sendTextOnly(text);
    }
}

async function sendTextOnly(text) {
    if (!pc) {
        await startWebRTC();
        await negotiate();
        // Wait for DC to open
        await new Promise(resolve => {
            if (dc.readyState === 'open') resolve();
            else dc.onopen = resolve;
        });
    }

    if (dc && dc.readyState === 'open') {
        dc.send(JSON.stringify({
            type: 'text_only',
            text: text
        }));
    }
}

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.textContent = role === 'system' ? 'AI' : 'U';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.textContent = text;

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(contentDiv);

    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
