const micBtn = document.getElementById('mic-btn');
const stopBtn = document.getElementById('stop-btn');
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
        sdpSemantics: 'unified-plan',
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
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

        // Ignore tokens from old generations
        if (data.generationId && data.generationId !== streamState.currentGenerationId) {
            console.log(`Ignoring token from old generation ${data.generationId} (current: ${streamState.currentGenerationId})`);
            return;
        }

        if (data.type === 'token') {
            handleStreamToken(data.text);
        } else if (data.type === 'done') {
            finalizeStream();
        } else if (data.type === 'response') {
            // Legacy/Error fallback
            if (currentThinkingMsg) {
                currentThinkingMsg.remove();
                currentThinkingMsg = null;
            }
            appendMessage('system', data.text);
        }
    };
}

// Streaming State
let streamState = {
    isStreaming: false,
    messageDiv: null,
    contentDiv: null,
    thinkingDetails: null,
    thinkingContent: null,
    mainContent: null,
    inThinkingBlock: false,
    buffer: ''
};

function handleStreamToken(token) {
    // Remove initial thinking indicator if this is the start
    if (currentThinkingMsg) {
        currentThinkingMsg.remove();
        currentThinkingMsg = null;
    }

    // CRITICAL: If we're already streaming and this is a new message,
    // finalize the old one first to prevent mixing
    if (streamState.isStreaming && !streamState.messageDiv) {
        // Edge case: streaming flag is set but no message div
        finalizeStream();
    }

    if (!streamState.isStreaming) {
        // Initialize new message
        streamState.isStreaming = true;
        streamState.stopRequested = false;

        // Show stop button
        if (stopBtn) {
            stopBtn.classList.remove('hidden');
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.textContent = 'AI';

        streamState.contentDiv = document.createElement('div');
        streamState.contentDiv.className = 'content';

        // We'll append structured content here.
        // Initially, we just have a main text node or thinking box.
        // To keep it simple, we'll append text nodes to mainContent container if not thinking.
        streamState.mainContent = document.createElement('span');
        streamState.contentDiv.appendChild(streamState.mainContent);

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(streamState.contentDiv);

        messagesContainer.appendChild(msgDiv);
        streamState.messageDiv = msgDiv;
        streamState.buffer = '';
        streamState.inThinkingBlock = false;
    }

    // Accumulate buffer to detect tags
    streamState.buffer += token;

    // Check for tag transitions
    // 1. Enter thinking: <think>
    if (!streamState.inThinkingBlock && streamState.buffer.includes('<think>')) {
        const parts = streamState.buffer.split('<think>');
        const preText = parts[0];
        streamState.buffer = parts[1] || ''; // Remaining after tag

        if (preText) {
            streamState.mainContent.textContent += preText;
        }

        streamState.inThinkingBlock = true;

        // Create thinking box if needed
        if (!streamState.thinkingDetails) {
            streamState.thinkingDetails = document.createElement('details');
            streamState.thinkingDetails.className = 'thinking-box';

            const summary = document.createElement('summary');
            summary.textContent = 'Thinking Process';

            streamState.thinkingContent = document.createElement('div');
            streamState.thinkingContent.className = 'thinking-content';

            streamState.thinkingDetails.appendChild(summary);
            streamState.thinkingDetails.appendChild(streamState.thinkingContent);

            // Insert before main content usually
            streamState.contentDiv.insertBefore(streamState.thinkingDetails, streamState.mainContent);

            // Auto-expand while generating? Maybe.
            streamState.thinkingDetails.open = true;
        }
    }

    // 2. Exit thinking: </think>
    if (streamState.inThinkingBlock && streamState.buffer.includes('</think>')) {
        const parts = streamState.buffer.split('</think>');
        const thinkText = parts[0];
        streamState.buffer = parts[1] || '';

        if (thinkText) {
            streamState.thinkingContent.textContent += thinkText;
        }

        streamState.inThinkingBlock = false;
        // Close box when done?
        streamState.thinkingDetails.open = false;
    }

    // 3. Normal content processing
    // If we successfully processed tags, streamState.buffer contains the 'rest'.
    // If no tags found yet, we might be inside a tag or just normal text.
    // To be safe, we only append if we are sure we aren't splitting a tag.
    // Simple heuristic: if buffer ends with '<', wait.

    // Optimized: convert buffer to text immediately if no partial tag risk
    // Only risk is '<' at end.

    if (!streamState.buffer.includes('<')) {
        if (streamState.inThinkingBlock) {
            streamState.thinkingContent.textContent += streamState.buffer;
        } else {
            // Filter ChatML tags if they leak
            let cleanText = streamState.buffer.replace('<|im_start|>', '').replace('assistant', '');
            streamState.mainContent.textContent += cleanText;
        }
        streamState.buffer = '';
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function finalizeStream() {
    if (!streamState.isStreaming) return;

    // Flush remaining buffer
    if (streamState.buffer) {
        if (streamState.inThinkingBlock) {
            streamState.thinkingContent.textContent += streamState.buffer;
        } else {
            streamState.mainContent.textContent += streamState.buffer;
        }
    }

    // Save to history
    const text = streamState.contentDiv ? streamState.contentDiv.innerText : ""; // Get visible text (approx)
    if (currentMessages.length > 0) {
        currentMessages.push({ role: 'assistant', text: text });
        saveCurrentChat();
    }

    streamState.isStreaming = false;
    streamState.messageDiv = null;
    streamState.thinkingDetails = null;
    streamState.buffer = '';
}

async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    console.log('ICE gathering started...');
    // Wait for ICE gathering to complete to ensure all candidates are included
    if (pc.iceGatheringState !== 'complete') {
        await new Promise(resolve => {
            const checkState = () => {
                if (pc.iceGatheringState === 'complete') {
                    pc.removeEventListener('icegatheringstatechange', checkState);
                    console.log('ICE gathering complete.');
                    resolve();
                }
            };
            pc.addEventListener('icegatheringstatechange', checkState);
            // Fallback timeout in case it never completes (e.g. no network)
            setTimeout(() => {
                console.warn('ICE gathering timed out, sending what we have.');
                resolve();
            }, 3000);
        });
    } else {
        console.log('ICE gathering already complete.');
    }

    console.log('Sending offer to server...');
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

    if (!response.ok) {
        throw new Error('Server returned error: ' + response.statusText);
    }

    const answer = await response.json();
    console.log('Received answer from server. Setting remote description...');
    await pc.setRemoteDescription(answer);
    console.log('WebRTC negotiation complete.');
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

if (stopBtn) {
    stopBtn.addEventListener('click', () => {
        if (!streamState.isStreaming) return;

        console.log('Stop requested by user.');
        streamState.stopRequested = true;

        // Send stop signal to server
        if (dc && dc.readyState === 'open') {
            dc.send(JSON.stringify({ type: 'stop' }));
        }

        finalizeStream();
        stopBtn.classList.add('hidden');
    });
}

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
    // Auto-stop any active generation before starting recording
    if (streamState.isStreaming) {
        console.log('Stopping active generation before starting recording');
        finalizeStream();
    }

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
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
            alert('Microphone access denied. Please allow microphone permissions.');
        } else if (location.hostname !== 'localhost' && location.protocol === 'http:') {
            alert('Microphone access requires a secure connection (HTTPS). You are using HTTP. Please setup HTTPS or use localhost.');
        } else {
            alert('Could not access microphone. Error: ' + e.message);
        }
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
        console.log('Sending stop message to server');
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
        console.log('Starting WebRTC connection...');
        await startWebRTC();
        console.log('Negotiating...');
        await negotiate();
        console.log('Waiting for DataChannel to open...');
        // Wait for DC to open with timeout
        try {
            await new Promise((resolve, reject) => {
                if (dc.readyState === 'open') {
                    resolve();
                } else {
                    const onOpen = () => {
                        cleanup();
                        resolve();
                    };
                    const onError = (e) => {
                        cleanup();
                        reject(new Error('DataChannel error: ' + e));
                    };
                    const timeoutId = setTimeout(() => {
                        cleanup();
                        reject(new Error('DataChannel connection timed out'));
                    }, 10000); // 10 second timeout

                    const cleanup = () => {
                        if (dc) {
                            dc.removeEventListener('open', onOpen);
                            dc.removeEventListener('error', onError);
                        }
                        clearTimeout(timeoutId);
                    };

                    dc.addEventListener('open', onOpen);
                    dc.addEventListener('error', onError);
                }
            });
            console.log('DataChannel opened successfully.');
        } catch (e) {
            console.error('Connection failed:', e);
            statusIndicator.textContent = 'Connection Timeout';
            appendMessage('system', 'Error: Could not connect to server. Please try refreshing.');
            return;
        }
    }

    if (dc && dc.readyState === 'open') {
        dc.send(JSON.stringify({
            type: 'text_only',
            text: text,
            generationId: streamState.currentGenerationId
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

    // Check for thinking tags
    // Pattern: <think> ... </think>
    // We handle the case where content might be mixed or multiple blocks, 
    // but typically it's one block at the start.

    // Simple regex for extracting think block
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
    let match;
    let lastIndex = 0;
    let hasThinking = false;

    // We'll build the content nodes
    const contentFragment = document.createDocumentFragment();

    while ((match = thinkRegex.exec(text)) !== null) {
        hasThinking = true;

        // Add text before the think block
        const beforeText = text.substring(lastIndex, match.index);
        if (beforeText) {
            contentFragment.appendChild(document.createTextNode(beforeText));
        }

        // Add the thinking box
        const thoughts = match[1];
        const details = document.createElement('details');
        details.className = 'thinking-box';
        const summary = document.createElement('summary');
        summary.textContent = 'Thinking Process';
        const p = document.createElement('div');
        p.className = 'thinking-content';
        p.textContent = thoughts; // Use textContent to avoid XSS

        details.appendChild(summary);
        details.appendChild(p);
        contentFragment.appendChild(details);

        lastIndex = thinkRegex.lastIndex;
    }

    // Add remaining text
    const remainingText = text.substring(lastIndex);
    if (remainingText) {
        contentFragment.appendChild(document.createTextNode(remainingText));
    }

    if (hasThinking) {
        contentDiv.appendChild(contentFragment);
    } else {
        contentDiv.textContent = text;
    }

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
