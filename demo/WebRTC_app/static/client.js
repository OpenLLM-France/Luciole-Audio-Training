const micBtn = document.getElementById('mic-btn');
const sendBtn = document.getElementById('send-btn');
const textInput = document.getElementById('text-input');
const messagesContainer = document.getElementById('messages');
const statusIndicator = document.getElementById('connection-status');
const newChatBtn = document.querySelector('.new-chat-btn');
const historyContainer = document.querySelector('.history');

let pc = null;
let dc = null;
let localStream = null;
let isRecording = false;
let currentFile = null;
let currentThinkingMsg = null; // Track thinking indicator
let sessionId = 'session_' + Date.now(); // Generate unique session ID
let currentMessages = []; // Track current chat messages
const filePreview = document.getElementById('file-preview');
const fileNameSpan = document.getElementById('file-name');
const removeFileBtn = document.getElementById('remove-file-btn');

// Chat History Management
const STORAGE_KEY = 'salem_chat_history';

function loadChatHistory() {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : { chats: [], currentChatId: null };
}

function saveChatHistory(history) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function saveCurrentChat() {
    if (currentMessages.length === 0) return; // Don't save empty chats

    const history = loadChatHistory();
    const chatTitle = currentMessages[0]?.text?.substring(0, 50) || 'New Chat';

    // Check if chat already exists
    const existingIndex = history.chats.findIndex(c => c.id === sessionId);
    const chatData = {
        id: sessionId,
        title: chatTitle,
        timestamp: Date.now(),
        messages: currentMessages
    };

    if (existingIndex >= 0) {
        history.chats[existingIndex] = chatData;
    } else {
        history.chats.unshift(chatData); // Add to beginning
    }

    saveChatHistory(history);
    renderChatHistory();
}

function loadChat(chatId) {
    const history = loadChatHistory();
    const chat = history.chats.find(c => c.id === chatId);

    if (!chat) return;

    // Save current chat before switching
    saveCurrentChat();

    // Load the selected chat
    sessionId = chat.id;
    currentMessages = [...chat.messages]; // Clone the messages array

    // Clear and restore messages WITHOUT triggering auto-save
    messagesContainer.innerHTML = '';

    // Temporarily disable auto-save during restore
    const tempMessages = currentMessages;
    currentMessages = []; // Prevent auto-save during render

    tempMessages.forEach(msg => {
        appendMessage(msg.role, msg.text);
    });

    // Restore the messages array
    currentMessages = tempMessages;

    // Update current chat ID
    history.currentChatId = chatId;
    saveChatHistory(history);
    renderChatHistory();
}

function renderChatHistory() {
    const history = loadChatHistory();
    historyContainer.innerHTML = '';

    history.chats.forEach(chat => {
        const item = document.createElement('div');
        item.className = 'history-item';
        if (chat.id === sessionId) {
            item.classList.add('active');
        }

        const date = new Date(chat.timestamp);
        const timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        item.innerHTML = `
            <div class="history-content">
                <div class="history-title">${chat.title}</div>
                <div class="history-date">${timeStr}</div>
            </div>
            <div class="history-actions">
                <button class="history-btn rename-btn" title="Rename">✏️</button>
                <button class="history-btn delete-btn" title="Delete">🗑️</button>
            </div>
        `;

        // Click to load chat (only on main area, not buttons)
        item.querySelector('.history-content').addEventListener('click', () => loadChat(chat.id));

        // Rename button
        item.querySelector('.rename-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            renameChat(chat.id);
        });

        // Delete button
        item.querySelector('.delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteChat(chat.id);
        });

        historyContainer.appendChild(item);
    });
}

function deleteChat(chatId) {
    if (!confirm('Delete this chat?')) return;

    const history = loadChatHistory();
    history.chats = history.chats.filter(c => c.id !== chatId);

    // If deleting current chat, start new one
    if (chatId === sessionId) {
        sessionId = 'session_' + Date.now();
        currentMessages = [];
        messagesContainer.innerHTML = '';
        // Re-add welcome message
        const welcomeMsg = document.createElement('div');
        welcomeMsg.className = 'message system';
        welcomeMsg.innerHTML = `
            <div class="avatar">AI</div>
            <div class="content">Hello! I am SALeM. You can chat with me using text or voice.</div>
        `;
        messagesContainer.appendChild(welcomeMsg);
    }

    saveChatHistory(history);
    renderChatHistory();
}

function renameChat(chatId) {
    const history = loadChatHistory();
    const chat = history.chats.find(c => c.id === chatId);
    if (!chat) return;

    const newTitle = prompt('Enter new title:', chat.title);
    if (newTitle && newTitle.trim()) {
        chat.title = newTitle.trim();
        saveChatHistory(history);
        renderChatHistory();
    }
}


// New Chat button handler
newChatBtn.addEventListener('click', () => {
    // Save current chat before starting new one
    saveCurrentChat();

    // Generate new session ID
    sessionId = 'session_' + Date.now();
    currentMessages = [];

    // Clear messages (keep only the welcome message)
    const welcomeMsg = messagesContainer.querySelector('.message.system');
    messagesContainer.innerHTML = '';
    if (welcomeMsg) {
        messagesContainer.appendChild(welcomeMsg);
    }

    // Reset connection
    if (pc) {
        pc.close();
        pc = null;
    }
    dc = null;
    statusIndicator.textContent = 'Disconnected';
    statusIndicator.classList.remove('connected');

    // Clear any file selection
    currentFile = null;
    filePreview.classList.add('hidden');

    // Update UI
    const history = loadChatHistory();
    history.currentChatId = sessionId;
    saveChatHistory(history);
    renderChatHistory();

    console.log('New chat started with session:', sessionId);
});

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
        statusIndicator.textContent = 'Recording Ready';
        statusIndicator.classList.add('connected');
    };

    channel.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        if (data.type === 'response') {
            // Remove thinking indicator if exists
            if (currentThinkingMsg) {
                currentThinkingMsg.remove();
                currentThinkingMsg = null;
            }
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
            type: pc.localDescription.type,
            sessionId: sessionId
        })
    });

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
}

// UI Interactions
function autoResizeTextarea() {
    textInput.style.height = 'auto';
    textInput.style.height = Math.min(textInput.scrollHeight, 200) + 'px';
}

textInput.addEventListener('input', autoResizeTextarea);
textInput.addEventListener('paste', () => setTimeout(autoResizeTextarea, 0));

textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

removeFileBtn.addEventListener('click', () => {
    currentFile = null;
    audioInput.value = '';
    filePreview.classList.add('hidden');
});

const uploadBtn = document.getElementById('upload-btn');
const audioInput = document.getElementById('audio-upload');

uploadBtn.addEventListener('click', () => {
    audioInput.click();
});

audioInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    fileNameSpan.textContent = file.name;
    filePreview.classList.remove('hidden');
});

async function uploadFile(file, prompt) {
    console.log('uploadFile function called with:', file, prompt);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('sessionId', sessionId);
    if (prompt) {
        formData.append('text', prompt);
    }

    console.log('Sending POST to /upload...');

    // Show thinking indicator
    const thinkingMsg = appendMessage('system', '');
    const thinkingIndicator = document.createElement('div');
    thinkingIndicator.className = 'thinking-indicator';
    thinkingIndicator.innerHTML = '<span></span><span></span><span></span>';
    thinkingMsg.querySelector('.content').appendChild(thinkingIndicator);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status);

        // Remove thinking indicator
        thinkingMsg.remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage('system', data.text);
        } else {
            let errorText = 'Error uploading file.';
            try {
                const errorData = await response.text();
                // Try to parse if it's JSON, otherwise use text
                try {
                    const jsonErr = JSON.parse(errorData);
                    if (jsonErr.text) errorText = jsonErr.text;
                    else if (jsonErr.message) errorText = jsonErr.message;
                } catch (e) {
                    if (errorData) errorText = `Error: ${errorData}`;
                }
            } catch (e) { }
            appendMessage('system', errorText);
        }
    } catch (e) {
        console.error('Upload error:', e);
        thinkingMsg.remove();
        appendMessage('system', 'Error uploading file.');
    }
}

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

        // Show thinking indicator
        currentThinkingMsg = appendMessage('system', '');
        const thinkingIndicator = document.createElement('div');
        thinkingIndicator.className = 'thinking-indicator';
        thinkingIndicator.innerHTML = '<span></span><span></span><span></span>';
        currentThinkingMsg.querySelector('.content').appendChild(thinkingIndicator);
    }

    // Do NOT close the PC here, otherwise we won't receive the response!
    // We keep the connection open for the session.
}

function sendMessage() {
    const text = textInput.value.trim();

    console.log('sendMessage called, currentFile:', currentFile, 'text:', text);

    if (currentFile) {
        // Handle file upload with optional text
        console.log('Uploading file:', currentFile.name, 'with prompt:', text);
        appendMessage('user', `Uploaded: ${currentFile.name} ${text ? `\nPrompt: ${text}` : ''}`);
        uploadFile(currentFile, text);

        // Reset state
        currentFile = null;
        filePreview.classList.add('hidden');
        audioInput.value = '';
        textInput.value = '';
        textInput.style.height = 'auto';
        return;
    }

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

        // Show thinking indicator
        currentThinkingMsg = appendMessage('system', '');
        const thinkingIndicator = document.createElement('div');
        thinkingIndicator.className = 'thinking-indicator';
        thinkingIndicator.innerHTML = '<span></span><span></span><span></span>';
        currentThinkingMsg.querySelector('.content').appendChild(thinkingIndicator);
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

    // Track message in current chat (skip welcome message)
    if (currentMessages.length > 0 || role === 'user') {
        currentMessages.push({ role, text });
        // Auto-save after each message
        saveCurrentChat();
    }

    return msgDiv;
}

// Initialize on page load
renderChatHistory();
