// =============================================================================
// SALeM Chat Client
// =============================================================================

// ---- DOM references ---------------------------------------------------------
const micBtn = document.getElementById('mic-btn');
const stopBtn = document.getElementById('stop-btn');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const audioInput = document.getElementById('audio-upload');
const textInput = document.getElementById('text-input');
const messagesContainer = document.getElementById('messages');
const messagesScroll = document.getElementById('messages-scroll');
const statusIndicator = document.getElementById('connection-status');
const newChatBtn = document.getElementById('new-chat-btn');
const historyContainer = document.getElementById('history');
const filePreview = document.getElementById('file-preview');
const fileNameSpan = document.getElementById('file-name');
const removeFileBtn = document.getElementById('remove-file-btn');
const themeToggle = document.getElementById('theme-toggle');
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.getElementById('sidebar');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const modalOverlay = document.getElementById('modal-overlay');
const modalTitle = document.getElementById('modal-title');
const modalDescription = document.getElementById('modal-description');
const modalInput = document.getElementById('modal-input');
const modalCancel = document.getElementById('modal-cancel');
const modalConfirm = document.getElementById('modal-confirm');
const toast = document.getElementById('toast');
const settingsBtn = document.getElementById('settings-btn');
const settingsOverlay = document.getElementById('settings-overlay');
const settingsClose = document.getElementById('settings-close');
const settingsReset = document.getElementById('settings-reset');
const setSilenceMs = document.getElementById('set-silence-ms');
const setSpeechThreshold = document.getElementById('set-speech-threshold');
const setMaxTokens = document.getElementById('set-max-tokens');
const setInstruction = document.getElementById('set-instruction');
const micMeter = document.getElementById('mic-meter');
const micMeterBar = micMeter ? micMeter.querySelector('.mic-meter-bar') : null;
const scrollBottomBtn = document.getElementById('scroll-bottom-btn');
const dropOverlay = document.getElementById('drop-overlay');
const chatArea = document.getElementById('chat-area');
const suggestedPromptsTpl = document.getElementById('suggested-prompts-template');

// ---- State ------------------------------------------------------------------
let pc = null;
let dc = null;
let localStream = null;
let isRecording = false;
let currentFile = null;
let currentThinkingMsg = null;
let sessionId = 'session_' + Date.now();
let currentMessages = [];
let pttHoldActive = false;
let reconnectAttempt = 0;
let reconnectTimer = null;
let streamStartTime = 0;
let streamTokenCount = 0;

// Per-recording: a MediaRecorder that captures the same mic stream we send
// over WebRTC, so we can replay the user's audio locally.
let mediaRecorder = null;
let recordedChunks = [];
let pendingMicAudioMsg = null; // user message div awaiting its audio blob

// Track of the most-recent user audio message that's waiting on a transcript
// from the server (mic) or the /transcribe endpoint (upload).
let pendingTranscriptMsg = null;

const streamState = {
    isStreaming: false,
    messageDiv: null,
    contentDiv: null,
    thinkingDetails: null,
    thinkingContent: null,
    mainContentText: '',
    mainContentEl: null,
    inThinkingBlock: false,
    buffer: '',
    currentGenerationId: 0,   // last gen ID we *sent* to the server
    activeGenerationId: null, // gen ID of the message currently being rendered
    stopRequested: false,
};

const STORAGE_KEY = 'salem_chat_history';
const THEME_KEY = 'salem_theme';
const SETTINGS_KEY = 'salem_settings';

const DEFAULT_SETTINGS = {
    micMode: 'vad',          // 'vad' | 'ptt'
    silenceMs: 1500,
    speechThreshold: 0.025,
    maxTokens: 256,
    instruction: 'Listen to the audio and answer the question:',
    effortMode: 'normal',    // 'normal' | 'max' — server pins decoding params when 'max'
};

// VAD tunables (silenceMs and speechThreshold are read from settings)
const VAD_BASE = {
    silenceThresholdRatio: 0.6, // hysteresis: silence threshold = speech * this
    minSpeechMs: 250,
    maxRecordingMs: 30000,
    pollIntervalMs: 50,
};

let vadController = null;

// =============================================================================
// Settings (localStorage)
// =============================================================================
function loadSettings() {
    try {
        const raw = localStorage.getItem(SETTINGS_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return { ...DEFAULT_SETTINGS, ...parsed };
    } catch (e) {
        return { ...DEFAULT_SETTINGS };
    }
}

function saveSettings(settings) {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (e) { /* ignore */ }
}

let settings = loadSettings();

function applyMicModeUI() {
    if (settings.micMode === 'ptt') {
        micBtn.classList.add('ptt-mode');
        micBtn.title = 'Hold to record';
    } else {
        micBtn.classList.remove('ptt-mode');
        micBtn.title = 'Record audio';
    }
}

// =============================================================================
// Theme (6 variants)
// =============================================================================
const THEMES = [
    { id: 'linagora', name: 'Linagora', meta: 'White + red', mode: 'light', bubble: '#d50000', bg: '#ffffff', border: '#dde1e8' },
    { id: 'carbon', name: 'Carbon', meta: 'Black + green', mode: 'dark', bubble: '#22c55e', bg: '#0e0f12', border: '#262830' },
    { id: 'paper', name: 'Paper', meta: 'Warm light', mode: 'light', bubble: '#b9612b', bg: '#f4ede0', border: '#dcd2c0' },
    { id: 'midnight', name: 'Midnight', meta: 'Deep dark', mode: 'dark', bubble: '#e8a13b', bg: '#15171f', border: '#2a2c34' },
    { id: 'terminal', name: 'Terminal', meta: 'CRT phosphor', mode: 'dark', bubble: '#f0a93b', bg: '#070a07', border: '#1a2418' },
    { id: 'ocean', name: 'Ocean', meta: 'Deep teal', mode: 'dark', bubble: '#e89154', bg: '#0e2227', border: '#234049' },
];
const THEME_IDS = THEMES.map((t) => t.id);

function themeMode(themeId) {
    const t = THEMES.find((x) => x.id === themeId);
    return t ? t.mode : 'light';
}

// Remember the user's most-recently-picked theme in each mode so the
// dark/light toggle flips back to whatever they were using before.
const LAST_LIGHT_KEY = 'salem_last_light';
const LAST_DARK_KEY = 'salem_last_dark';
function loadLast(key, fallback) {
    try {
        const v = localStorage.getItem(key);
        return THEME_IDS.includes(v) ? v : fallback;
    } catch (e) { return fallback; }
}
let lastLightTheme = loadLast(LAST_LIGHT_KEY, 'linagora');
let lastDarkTheme = loadLast(LAST_DARK_KEY, 'carbon');

function applyTheme(theme) {
    if (!THEME_IDS.includes(theme)) theme = 'linagora';
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* ignore */ }
    const lightCss = document.getElementById('hljs-theme-light');
    const darkCss = document.getElementById('hljs-theme-dark');
    const isDark = themeMode(theme) === 'dark';
    if (lightCss && darkCss) {
        lightCss.disabled = isDark;
        darkCss.disabled = !isDark;
    }
    if (isDark) {
        lastDarkTheme = theme;
        try { localStorage.setItem(LAST_DARK_KEY, theme); } catch (e) { /* ignore */ }
    } else {
        lastLightTheme = theme;
        try { localStorage.setItem(LAST_LIGHT_KEY, theme); } catch (e) { /* ignore */ }
    }
    renderThemePicker(theme);
}

function toggleLightDark() {
    const current = document.documentElement.getAttribute('data-theme') || 'linagora';
    const isDark = themeMode(current) === 'dark';
    applyTheme(isDark ? lastLightTheme : lastDarkTheme);
}

themeToggle.addEventListener('click', toggleLightDark);

function renderThemePicker(activeId) {
    const picker = document.getElementById('theme-picker');
    if (!picker) return;
    picker.innerHTML = '';
    THEMES.forEach((t) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'theme-swatch' + (t.id === activeId ? ' active' : '');
        btn.dataset.theme = t.id;
        btn.setAttribute('aria-label', t.name + ' theme');
        btn.innerHTML = `
            <span class="theme-swatch-preview" style="background:${t.bg};border-color:${t.border};">
                <span class="theme-swatch-bubble" style="background:${t.bubble}"></span>
                <span class="theme-swatch-line" style="background:${t.bubble}"></span>
            </span>
            <span class="theme-swatch-name">${t.name}</span>
            <span class="theme-swatch-meta">${t.meta}</span>
        `;
        btn.addEventListener('click', () => applyTheme(t.id));
        picker.appendChild(btn);
    });
}

// Keep hljs in sync with whatever data-theme is set on boot.
applyTheme(document.documentElement.getAttribute('data-theme') || 'linagora');

// =============================================================================
// Markdown rendering
// =============================================================================
if (window.marked) {
    marked.setOptions({ gfm: true, breaks: false });
}

function renderMarkdown(text) {
    if (!window.marked || !window.DOMPurify) {
        const p = document.createElement('p');
        p.textContent = text;
        return p.outerHTML;
    }
    const html = marked.parse(text);
    const sanitized = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    if (!window.hljs) return sanitized;
    const tmp = document.createElement('div');
    tmp.innerHTML = sanitized;
    tmp.querySelectorAll('pre code').forEach((block) => {
        const cls = block.className || '';
        const langMatch = cls.match(/language-([\w-]+)/);
        try {
            if (langMatch && hljs.getLanguage(langMatch[1])) {
                const result = hljs.highlight(block.textContent, { language: langMatch[1] });
                block.innerHTML = result.value;
            } else {
                const result = hljs.highlightAuto(block.textContent);
                block.innerHTML = result.value;
            }
            block.classList.add('hljs');
        } catch (e) { /* leave as-is */ }
    });
    return tmp.innerHTML;
}

// =============================================================================
// Modal (replacement for window.prompt / window.confirm)
// =============================================================================
let modalResolver = null;

function openModal({ title, description = '', defaultValue = null, confirmText = 'OK', cancelText = 'Cancel', destructive = false }) {
    return new Promise((resolve) => {
        modalResolver = resolve;
        modalTitle.textContent = title;
        modalDescription.textContent = description;
        modalConfirm.textContent = confirmText;
        modalCancel.textContent = cancelText;
        modalConfirm.classList.toggle('btn-primary', !destructive);
        modalConfirm.classList.toggle('btn-destructive', destructive);

        if (defaultValue !== null) {
            modalInput.classList.remove('hidden');
            modalInput.value = defaultValue;
        } else {
            modalInput.classList.add('hidden');
            modalInput.value = '';
        }

        modalOverlay.classList.remove('hidden');

        if (defaultValue !== null) {
            requestAnimationFrame(() => { modalInput.focus(); modalInput.select(); });
        } else {
            requestAnimationFrame(() => modalConfirm.focus());
        }
    });
}

function closeModal(result) {
    modalOverlay.classList.add('hidden');
    if (modalResolver) {
        modalResolver(result);
        modalResolver = null;
    }
}

modalConfirm.addEventListener('click', () => {
    const result = modalInput.classList.contains('hidden') ? true : modalInput.value.trim();
    closeModal(result);
});
modalCancel.addEventListener('click', () => closeModal(null));
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal(null);
});
modalInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); modalConfirm.click(); }
    else if (e.key === 'Escape') { closeModal(null); }
});

// =============================================================================
// Settings panel
// =============================================================================
function refreshSegmented(scope, settingName, value) {
    const group = scope.querySelector(`.settings-segmented[data-setting="${settingName}"]`);
    if (!group) return;
    group.querySelectorAll('button').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.value === value);
    });
}

function syncSettingsForm() {
    const themeNow = document.documentElement.getAttribute('data-theme') || 'linagora';
    renderThemePicker(themeNow);
    refreshSegmented(settingsOverlay, 'micMode', settings.micMode);
    refreshSegmented(settingsOverlay, 'effortMode', settings.effortMode);
    setSilenceMs.value = settings.silenceMs;
    setSpeechThreshold.value = settings.speechThreshold;
    setMaxTokens.value = settings.maxTokens;
    setInstruction.value = settings.instruction;
}

function openSettings() {
    syncSettingsForm();
    settingsOverlay.classList.remove('hidden');
    requestAnimationFrame(() => settingsClose.focus());
}
function closeSettings() {
    settingsOverlay.classList.add('hidden');
}
settingsBtn.addEventListener('click', openSettings);
settingsClose.addEventListener('click', closeSettings);
settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
});

settingsReset.addEventListener('click', () => {
    settings = { ...DEFAULT_SETTINGS };
    saveSettings(settings);
    syncSettingsForm();
    applyMicModeUI();
    showToast('Settings reset');
});

settingsOverlay.querySelectorAll('.settings-segmented').forEach((group) => {
    group.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-value]');
        if (!btn) return;
        const setting = group.dataset.setting;
        const value = btn.dataset.value;
        if (setting === 'micMode') {
            settings.micMode = value;
            saveSettings(settings);
            refreshSegmented(settingsOverlay, 'micMode', value);
            applyMicModeUI();
        } else if (setting === 'effortMode') {
            settings.effortMode = value;
            saveSettings(settings);
            applyEffortUI();
        }
    });
});

[setSilenceMs, setSpeechThreshold, setMaxTokens, setInstruction].forEach((input) => {
    input.addEventListener('change', () => {
        settings.silenceMs = clampNumber(setSilenceMs.value, 300, 5000, DEFAULT_SETTINGS.silenceMs);
        settings.speechThreshold = clampNumber(setSpeechThreshold.value, 0.005, 0.1, DEFAULT_SETTINGS.speechThreshold);
        settings.maxTokens = clampNumber(setMaxTokens.value, 16, 2048, DEFAULT_SETTINGS.maxTokens);
        settings.instruction = setInstruction.value || DEFAULT_SETTINGS.instruction;
        saveSettings(settings);
    });
});

function clampNumber(v, min, max, fallback) {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
}

// MediaRecorder blobs (especially WebM) often report duration === Infinity
// because the container has no Duration field. The standard workaround:
// once metadata is loaded, seek past the end; the browser then computes
// the actual duration, which we restore by seeking back to 0.
function fixAudioDuration(audioEl) {
    if (!audioEl) return;
    const onMeta = () => {
        if (Number.isFinite(audioEl.duration)) return;
        const onUpdate = () => {
            audioEl.removeEventListener('timeupdate', onUpdate);
            try { audioEl.currentTime = 0; } catch (e) { /* ignore */ }
        };
        audioEl.addEventListener('timeupdate', onUpdate);
        try { audioEl.currentTime = 1e9; } catch (e) { /* ignore */ }
    };
    if (audioEl.readyState >= 1) onMeta();
    else audioEl.addEventListener('loadedmetadata', onMeta, { once: true });
}

// =============================================================================
// Custom audio player (waveform-based, replaces native <audio controls>)
// =============================================================================
const WAVE_BAR_COUNT = 56;

// Decode the audio blob and reduce it to N peak amplitudes. Returns null on
// failure (unsupported codec, broken blob, etc.) so the caller can fall back
// to a flat track.
async function computeWaveformPeaks(blobUrl, bucketCount = WAVE_BAR_COUNT) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    let ctx;
    try {
        const res = await fetch(blobUrl);
        const arr = await res.arrayBuffer();
        ctx = new AudioCtx();
        const audioBuffer = await ctx.decodeAudioData(arr);
        const ch = audioBuffer.getChannelData(0);
        const bucketSize = Math.max(1, Math.floor(ch.length / bucketCount));
        const peaks = new Array(bucketCount);
        let max = 0;
        for (let i = 0; i < bucketCount; i++) {
            const start = i * bucketSize;
            const end = Math.min(ch.length, start + bucketSize);
            let m = 0;
            for (let j = start; j < end; j++) {
                const v = Math.abs(ch[j]);
                if (v > m) m = v;
            }
            peaks[i] = m;
            if (m > max) max = m;
        }
        // Normalize 0..1 with a non-linear curve so quiet tracks still
        // have visible bars (sqrt brings up the low end).
        if (max <= 0) return peaks;
        return peaks.map((p) => Math.sqrt(p / max));
    } catch (e) {
        console.warn('Waveform decode failed:', e);
        return null;
    } finally {
        if (ctx) try { await ctx.close(); } catch (e) { /* ignore */ }
    }
}

function buildAudioPlayer(srcUrl) {
    const audio = new Audio();
    audio.preload = 'metadata';
    audio.src = srcUrl;
    fixAudioDuration(audio);

    const wrap = document.createElement('div');
    wrap.className = 'audio-player';
    wrap.dataset.state = 'paused';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'audio-player-btn';
    btn.setAttribute('aria-label', 'Play');
    btn.innerHTML =
        '<svg class="icon-play" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6 4 20 12 6 20 6 4"></polygon></svg>' +
        '<svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="4" width="4" height="16" rx="1"></rect><rect x="14" y="4" width="4" height="16" rx="1"></rect></svg>';

    const wave = document.createElement('div');
    wave.className = 'audio-player-waveform';
    wave.setAttribute('role', 'slider');
    wave.setAttribute('aria-label', 'Seek');

    const timeEl = document.createElement('span');
    timeEl.className = 'audio-player-time';
    timeEl.textContent = '0:00 / 0:00';

    const speedBtn = document.createElement('button');
    speedBtn.type = 'button';
    speedBtn.className = 'audio-player-speed';
    speedBtn.textContent = '1×';
    speedBtn.setAttribute('aria-label', 'Playback speed');

    wrap.append(btn, wave, timeEl, speedBtn);

    let bars = [];
    function buildBars(peaks) {
        wave.innerHTML = '';
        bars = peaks.map((p) => {
            const bar = document.createElement('span');
            bar.className = 'audio-player-waveform-bar';
            // Heights between 3px and 22px so even silent buckets are visible.
            const h = 3 + Math.round(p * 19);
            bar.style.height = h + 'px';
            wave.appendChild(bar);
            return bar;
        });
        updateProgress();
    }

    // Initial flat placeholder while decoding.
    buildBars(new Array(WAVE_BAR_COUNT).fill(0.18));

    computeWaveformPeaks(srcUrl).then((peaks) => {
        if (peaks && peaks.length) buildBars(peaks);
    });

    function fmt(s) {
        if (!Number.isFinite(s) || s < 0) return '0:00';
        const total = Math.floor(s);
        const m = Math.floor(total / 60);
        const sec = String(total % 60).padStart(2, '0');
        return `${m}:${sec}`;
    }

    function updateProgress() {
        const dur = Number.isFinite(audio.duration) ? audio.duration : 0;
        if (dur > 0 && bars.length) {
            const ratio = Math.min(1, audio.currentTime / dur);
            const playedCount = Math.floor(ratio * bars.length);
            for (let i = 0; i < bars.length; i++) {
                bars[i].classList.toggle('played', i < playedCount);
            }
        }
        timeEl.textContent = `${fmt(audio.currentTime)} / ${fmt(dur)}`;
    }

    audio.addEventListener('loadedmetadata', updateProgress);
    audio.addEventListener('durationchange', updateProgress);
    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('seeked', updateProgress);
    audio.addEventListener('ended', () => {
        wrap.dataset.state = 'paused';
        btn.setAttribute('aria-label', 'Play');
        audio.currentTime = 0;
        updateProgress();
    });

    btn.addEventListener('click', () => {
        if (audio.paused) {
            audio.play().then(() => {
                wrap.dataset.state = 'playing';
                btn.setAttribute('aria-label', 'Pause');
            }).catch(() => { /* play denied */ });
        } else {
            audio.pause();
            wrap.dataset.state = 'paused';
            btn.setAttribute('aria-label', 'Play');
        }
    });

    function seekFromEvent(e) {
        const rect = wave.getBoundingClientRect();
        const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left)) / rect.width;
        const dur = Number.isFinite(audio.duration) ? audio.duration : 0;
        if (dur > 0) audio.currentTime = x * dur;
    }
    wave.addEventListener('click', seekFromEvent);

    const speeds = [1, 1.5, 2];
    let speedIdx = 0;
    speedBtn.addEventListener('click', () => {
        speedIdx = (speedIdx + 1) % speeds.length;
        audio.playbackRate = speeds[speedIdx];
        speedBtn.textContent = speeds[speedIdx] + '×';
    });

    updateProgress();
    return wrap;
}

function pickRecorderMime() {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
    if (typeof MediaRecorder === 'undefined') return null;
    for (const m of candidates) {
        if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
    }
    return null;
}

applyMicModeUI();

// ---- Inline effort picker (next to send button) ----
const effortPicker = document.querySelector('.effort-picker');
const effortToggle = document.getElementById('effort-toggle');
const effortToggleLabel = document.getElementById('effort-toggle-label');
const effortMenu = document.getElementById('effort-menu');

function applyEffortUI() {
    if (!effortPicker) return;
    const mode = settings.effortMode || 'normal';
    effortPicker.dataset.mode = mode;
    if (effortToggleLabel) {
        effortToggleLabel.textContent = mode === 'max' ? 'Max' : 'Normal';
    }
    if (effortMenu) {
        effortMenu.querySelectorAll('.effort-menu-item').forEach((item) => {
            item.classList.toggle('active', item.dataset.value === mode);
        });
    }
    refreshSegmented(settingsOverlay, 'effortMode', mode);
}

function closeEffortMenu() {
    if (effortMenu) effortMenu.classList.add('hidden');
    if (effortToggle) effortToggle.setAttribute('aria-expanded', 'false');
}

if (effortToggle && effortMenu) {
    effortToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const open = !effortMenu.classList.contains('hidden');
        if (open) closeEffortMenu();
        else {
            effortMenu.classList.remove('hidden');
            effortToggle.setAttribute('aria-expanded', 'true');
        }
    });
    effortMenu.addEventListener('click', (e) => {
        const item = e.target.closest('.effort-menu-item');
        if (!item) return;
        settings.effortMode = item.dataset.value;
        saveSettings(settings);
        applyEffortUI();
        closeEffortMenu();
    });
    document.addEventListener('click', (e) => {
        if (effortMenu.classList.contains('hidden')) return;
        if (!effortPicker.contains(e.target)) closeEffortMenu();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !effortMenu.classList.contains('hidden')) closeEffortMenu();
    });
}

applyEffortUI();

const clearContextBtn = document.getElementById('clear-context-btn');
if (clearContextBtn) {
    clearContextBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/reset-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId }),
            });
            if (res.ok) showToast('Context cleared');
            else showToast('Could not clear context');
        } catch (e) {
            showToast('Could not reach server');
        }
    });
}

// =============================================================================
// Toast
// =============================================================================
let toastTimeout = null;
function showToast(message, duration = 2400) {
    toast.textContent = message;
    toast.classList.remove('hidden');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.add('hidden'), duration);
}

// =============================================================================
// Status indicator
// =============================================================================
function setStatus(state, label) {
    statusIndicator.classList.remove('connected', 'connecting', 'error');
    if (state) statusIndicator.classList.add(state);
    statusIndicator.textContent = label;
}

// =============================================================================
// Auto-scroll (only if user is near bottom)
// =============================================================================
function isNearBottom(threshold = 80) {
    return messagesScroll.scrollHeight - messagesScroll.clientHeight - messagesScroll.scrollTop < threshold;
}
function maybeScrollToBottom() {
    if (isNearBottom(180)) {
        messagesScroll.scrollTop = messagesScroll.scrollHeight;
    } else {
        scrollBottomBtn.classList.remove('hidden');
    }
}
function forceScrollToBottom() {
    messagesScroll.scrollTop = messagesScroll.scrollHeight;
    scrollBottomBtn.classList.add('hidden');
}
messagesScroll.addEventListener('scroll', () => {
    if (isNearBottom(80)) scrollBottomBtn.classList.add('hidden');
});
scrollBottomBtn.addEventListener('click', forceScrollToBottom);

// =============================================================================
// Empty state / suggested prompts
// =============================================================================
function renderEmptyState() {
    messagesContainer.innerHTML = '';
    const node = suggestedPromptsTpl.content.cloneNode(true);
    messagesContainer.appendChild(node);
}

// =============================================================================
// Chat history (localStorage)
// =============================================================================
function loadChatHistory() {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : { chats: [], currentChatId: null };
    } catch (e) {
        return { chats: [], currentChatId: null };
    }
}

function saveChatHistory(history) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(history)); } catch (e) { /* ignore */ }
}

// Coalesce rapid-fire writes (token streaming, transcript arrival, audio
// blob attach) into one localStorage.setItem + one sidebar re-render per
// short window. Synchronous flushes happen on chat switch / new chat /
// page unload so nothing is lost.
let saveCurrentChatTimer = null;
function _doSaveCurrentChat() {
    saveCurrentChatTimer = null;
    if (currentMessages.length === 0) return;
    const history = loadChatHistory();
    const first = currentMessages[0] || {};
    const titleSource = (first.text && first.text.trim()) || (first.transcript && first.transcript.trim());
    const chatTitle = titleSource ? titleSource.substring(0, 50) : (first.audioUrl ? 'Voice message' : 'New Chat');
    const chatData = { id: sessionId, title: chatTitle, timestamp: Date.now(), messages: currentMessages };
    const existingIndex = history.chats.findIndex((c) => c.id === sessionId);
    if (existingIndex >= 0) history.chats[existingIndex] = chatData;
    else history.chats.unshift(chatData);
    saveChatHistory(history);
    renderChatHistory();
}
function saveCurrentChat({ flush = false } = {}) {
    if (flush) {
        if (saveCurrentChatTimer) { clearTimeout(saveCurrentChatTimer); saveCurrentChatTimer = null; }
        _doSaveCurrentChat();
        return;
    }
    if (saveCurrentChatTimer) return;
    saveCurrentChatTimer = setTimeout(_doSaveCurrentChat, 250);
}

function loadChat(chatId) {
    const history = loadChatHistory();
    const chat = history.chats.find((c) => c.id === chatId);
    if (!chat) return;

    saveCurrentChat({ flush: true });

    sessionId = chat.id;
    currentMessages = [...chat.messages];

    messagesContainer.innerHTML = '';
    const tempMessages = currentMessages;
    currentMessages = [];
    // Filter out empty assistant messages from older chats that pre-date the
    // finalizeStream guard. Done here rather than in saveCurrentChat so
    // existing localStorage data heals on next save.
    const cleaned = tempMessages.filter((msg) => {
        if (msg.role !== 'system') return true;
        return Boolean(msg.text && msg.text.trim());
    });
    cleaned.forEach((msg) => appendMessage(msg.role, msg.text, {
        silent: true,
        // Preserve the empty-string "pending" transcript shape; only omit when
        // the field was never set on this message.
        transcript: 'transcript' in msg ? msg.transcript : null,
        // Blob URLs survive in-session (no reload). They die on full reload —
        // IndexedDB would be needed to persist the actual bytes.
        audioUrl: msg.audioUrl || null,
    }));
    currentMessages = cleaned;

    history.currentChatId = chatId;
    saveChatHistory(history);
    renderChatHistory();
    forceScrollToBottom();
}

function renderChatHistory() {
    const history = loadChatHistory();
    historyContainer.innerHTML = '';

    if (history.chats.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'history-empty';
        empty.textContent = 'No conversations yet';
        historyContainer.appendChild(empty);
        return;
    }

    history.chats.forEach((chat) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        if (chat.id === sessionId) item.classList.add('active');

        const date = new Date(chat.timestamp);
        const timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        const content = document.createElement('div');
        content.className = 'history-content';
        const titleEl = document.createElement('div');
        titleEl.className = 'history-title';
        titleEl.textContent = chat.title;
        const dateEl = document.createElement('div');
        dateEl.className = 'history-date';
        dateEl.textContent = timeStr;
        content.appendChild(titleEl);
        content.appendChild(dateEl);

        const actions = document.createElement('div');
        actions.className = 'history-actions';

        const renameBtn = document.createElement('button');
        renameBtn.className = 'history-btn rename-btn';
        renameBtn.type = 'button';
        renameBtn.title = 'Rename';
        renameBtn.setAttribute('aria-label', 'Rename conversation');
        renameBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path></svg>';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'history-btn delete-btn';
        deleteBtn.type = 'button';
        deleteBtn.title = 'Delete';
        deleteBtn.setAttribute('aria-label', 'Delete conversation');
        deleteBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"></path><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path></svg>';

        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);

        item.appendChild(content);
        item.appendChild(actions);

        content.addEventListener('click', () => {
            loadChat(chat.id);
            closeSidebarMobile();
        });
        renameBtn.addEventListener('click', (e) => { e.stopPropagation(); renameChat(chat.id); });
        deleteBtn.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(chat.id); });

        historyContainer.appendChild(item);
    });
}

async function deleteChat(chatId) {
    const ok = await openModal({
        title: 'Delete conversation?',
        description: 'This action cannot be undone.',
        confirmText: 'Delete',
        destructive: true,
    });
    if (!ok) return;

    const history = loadChatHistory();
    const removed = history.chats.find((c) => c.id === chatId);
    history.chats = history.chats.filter((c) => c.id !== chatId);
    if (removed) revokeAudioUrls(removed.messages);

    if (chatId === sessionId) {
        revokeAudioUrls(currentMessages);
        sessionId = 'session_' + Date.now();
        currentMessages = [];
        renderEmptyState();
    }

    saveChatHistory(history);
    renderChatHistory();
    showToast('Conversation deleted');
}

async function renameChat(chatId) {
    const history = loadChatHistory();
    const chat = history.chats.find((c) => c.id === chatId);
    if (!chat) return;

    const newTitle = await openModal({
        title: 'Rename conversation',
        defaultValue: chat.title,
        confirmText: 'Save',
    });
    if (newTitle && typeof newTitle === 'string') {
        chat.title = newTitle.trim();
        saveChatHistory(history);
        renderChatHistory();
    }
}

newChatBtn.addEventListener('click', startNewChat);

function startNewChat() {
    saveCurrentChat({ flush: true });

    sessionId = 'session_' + Date.now();
    currentMessages = [];
    renderEmptyState();

    if (pc) { pc.close(); pc = null; }
    dc = null;
    setStatus(null, 'Idle');

    currentFile = null;
    filePreview.classList.add('hidden');

    const history = loadChatHistory();
    history.currentChatId = sessionId;
    saveChatHistory(history);
    renderChatHistory();
    closeSidebarMobile();
    textInput.focus();
}

// =============================================================================
// Sidebar (mobile toggle)
// =============================================================================
function openSidebarMobile() {
    sidebar.classList.add('open');
    sidebarBackdrop.classList.add('open');
}
function closeSidebarMobile() {
    sidebar.classList.remove('open');
    sidebarBackdrop.classList.remove('open');
}
function toggleSidebar() {
    if (sidebar.classList.contains('open')) closeSidebarMobile();
    else openSidebarMobile();
}
menuToggle.addEventListener('click', openSidebarMobile);
sidebarBackdrop.addEventListener('click', closeSidebarMobile);

// =============================================================================
// Voice Activity Detection (with optional level callback)
// =============================================================================
function startVAD(stream, { onSilence, onSpeechStart, onLevel, autoStop = true }) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) {
        console.warn('Web Audio API unavailable');
        return null;
    }

    const ctx = new AudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.4;
    source.connect(analyser);

    const data = new Float32Array(analyser.fftSize);
    const speechThreshold = settings.speechThreshold;
    const silenceThreshold = speechThreshold * VAD_BASE.silenceThresholdRatio;
    const silenceMs = settings.silenceMs;

    let armed = false;
    let speechStartedAt = 0;
    let silenceStartedAt = 0;
    const startedAt = performance.now();
    let stopped = false;

    function rms() {
        analyser.getFloatTimeDomainData(data);
        let s = 0;
        for (let i = 0; i < data.length; i++) s += data[i] * data[i];
        return Math.sqrt(s / data.length);
    }

    const intervalId = setInterval(() => {
        if (stopped) return;
        const r = rms();
        const now = performance.now();
        if (onLevel) onLevel(r, armed);

        if (autoStop && now - startedAt >= VAD_BASE.maxRecordingMs) {
            stop();
            onSilence && onSilence('max-duration');
            return;
        }
        if (!autoStop) return;

        if (!armed) {
            if (r >= speechThreshold) {
                if (speechStartedAt === 0) speechStartedAt = now;
                if (now - speechStartedAt >= VAD_BASE.minSpeechMs) {
                    armed = true; silenceStartedAt = 0;
                    onSpeechStart && onSpeechStart();
                }
            } else { speechStartedAt = 0; }
        } else {
            if (r < silenceThreshold) {
                if (silenceStartedAt === 0) silenceStartedAt = now;
                if (now - silenceStartedAt >= silenceMs) {
                    stop();
                    onSilence && onSilence('silence');
                }
            } else { silenceStartedAt = 0; }
        }
    }, VAD_BASE.pollIntervalMs);

    function stop() {
        if (stopped) return;
        stopped = true;
        clearInterval(intervalId);
        try { source.disconnect(); } catch (e) { /* ignore */ }
        try { ctx.close(); } catch (e) { /* ignore */ }
    }

    return { stop };
}

// =============================================================================
// Mic level meter
// =============================================================================
function showMeter() {
    if (micMeter) micMeter.classList.remove('hidden');
}
function hideMeter() {
    if (micMeter) micMeter.classList.add('hidden');
    if (micMeterBar) micMeterBar.style.width = '0%';
}
function updateMeter(rms, armed) {
    if (!micMeterBar) return;
    // Scale RMS (typically 0..0.3) to 0..100%
    const pct = Math.min(100, Math.round((rms / 0.2) * 100));
    micMeterBar.style.width = pct + '%';
    micMeterBar.classList.toggle('silent', rms < settings.speechThreshold * VAD_BASE.silenceThresholdRatio);
}

// =============================================================================
// WebRTC
// =============================================================================
async function startWebRTC() {
    const config = {
        sdpSemantics: 'unified-plan',
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    };

    pc = new RTCPeerConnection(config);
    setStatus('connecting', 'Connecting…');

    dc = pc.createDataChannel('chat');
    setupDataChannel(dc);

    pc.addEventListener('iceconnectionstatechange', () => {
        if (!pc) return;
        const s = pc.iceConnectionState;
        if (s === 'failed' || s === 'disconnected') {
            scheduleReconnect();
        }
    });

    return pc;
}

function setupDataChannel(channel) {
    channel.onopen = () => {
        setStatus('connected', 'Ready');
        reconnectAttempt = 0;
    };

    channel.onmessage = (evt) => {
        const data = JSON.parse(evt.data);

        if (
            data.generationId !== undefined &&
            data.generationId !== null &&
            streamState.currentGenerationId !== undefined &&
            data.generationId !== streamState.currentGenerationId
        ) {
            return; // stale
        }

        if (data.type === 'token') {
            handleStreamToken(data.text, data.generationId);
        } else if (data.type === 'done') {
            finalizeStream({ tokenCount: data.tokenCount, elapsedMs: data.elapsedMs });
        } else if (data.type === 'audio_transcript') {
            applyTranscript(data.text);
        } else if (data.type === 'queue_position') {
            handleQueuePosition(data);
        } else if (data.type === 'rejected') {
            handleRejected(data);
        } else if (data.type === 'response') {
            if (currentThinkingMsg) { currentThinkingMsg.remove(); currentThinkingMsg = null; }
            appendMessage('system', data.text);
        }
    };
}

function handleQueuePosition(data) {
    const pos = data.position;
    const depth = data.depth;
    if (typeof pos !== 'number') return;
    setStatus('connecting', `Queued · #${pos} of ${depth}`);
}

function handleRejected(data) {
    if (currentThinkingMsg) { currentThinkingMsg.remove(); currentThinkingMsg = null; }
    if (streamState.isStreaming) finalizeStream();
    if (stopBtn) stopBtn.classList.add('hidden');
    const reason = data.reason;
    if (reason === 'queue_full') {
        showToast('Server is busy — try again in a moment');
        setStatus('error', 'Queue full');
    } else if (reason === 'rate_limit') {
        showToast('Slow down — too many requests');
        setStatus('error', 'Rate limited');
    } else {
        showToast('Request rejected');
        setStatus('error', 'Rejected');
    }
}

function applyTranscript(text) {
    if (!pendingTranscriptMsg) return;
    const el = pendingTranscriptMsg._transcriptEl;
    if (el) {
        el.textContent = text;
        el.classList.remove('placeholder');
    }
    if (pendingTranscriptMsg._messageData) {
        pendingTranscriptMsg._messageData.transcript = text;
        saveCurrentChat();
    }
    pendingTranscriptMsg = null;
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    if (reconnectAttempt >= 5) {
        setStatus('error', 'Connection failed');
        return;
    }
    const delay = Math.min(16000, 1000 * Math.pow(2, reconnectAttempt));
    reconnectAttempt += 1;
    setStatus('connecting', `Reconnecting in ${Math.round(delay / 1000)}s…`);
    reconnectTimer = setTimeout(async () => {
        reconnectTimer = null;
        try {
            if (pc) { try { pc.close(); } catch (e) { } }
            pc = null; dc = null;
            await startWebRTC();
            await negotiate();
        } catch (e) {
            console.error('Reconnect failed:', e);
            scheduleReconnect();
        }
    }, delay);
}

// =============================================================================
// Streaming token handler (with <think> tag parsing)
// =============================================================================
function stripChatMLTags(text) {
    return text
        .replace(/<\|im_start\|>assistant\s*/g, '')
        .replace(/<\|im_start\|>/g, '')
        .replace(/<\|im_end\|>/g, '');
}

function appendMainContent(textChunk) {
    if (!textChunk) return;
    streamState.mainContentText += textChunk;
    if (streamState.mainContentEl) {
        streamState.mainContentEl.innerHTML = renderMarkdown(streamState.mainContentText);
    }
}

function handleStreamToken(token, incomingGenerationId) {
    if (currentThinkingMsg) { currentThinkingMsg.remove(); currentThinkingMsg = null; }

    // If we're already rendering a stream but this token belongs to a different
    // generation, finalize the old one first so this message gets its own bubble.
    if (
        streamState.isStreaming &&
        incomingGenerationId !== undefined &&
        streamState.activeGenerationId !== null &&
        streamState.activeGenerationId !== incomingGenerationId
    ) {
        finalizeStream();
    }

    if (streamState.isStreaming && !streamState.messageDiv) finalizeStream();

    if (!streamState.isStreaming) {
        streamState.isStreaming = true;
        streamState.stopRequested = false;
        streamState.mainContentText = '';
        streamState.activeGenerationId = incomingGenerationId !== undefined ? incomingGenerationId : streamState.currentGenerationId;
        streamStartTime = performance.now();
        streamTokenCount = 0;
        setStatus('connected', 'Generating…');
        if (stopBtn) stopBtn.classList.remove('hidden');

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.setAttribute('aria-hidden', 'true');
        avatarDiv.textContent = 'AI';

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';

        streamState.contentDiv = document.createElement('div');
        streamState.contentDiv.className = 'content';

        streamState.mainContentEl = document.createElement('div');
        streamState.mainContentEl.className = 'markdown-body';
        streamState.contentDiv.appendChild(streamState.mainContentEl);

        wrapper.appendChild(streamState.contentDiv);
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(wrapper);

        messagesContainer.appendChild(msgDiv);
        streamState.messageDiv = msgDiv;
        streamState.buffer = '';
        streamState.inThinkingBlock = false;
    }

    streamState.buffer += token;
    streamTokenCount += 1;

    // Enter thinking block
    if (!streamState.inThinkingBlock && streamState.buffer.includes('<think>')) {
        const parts = streamState.buffer.split('<think>');
        const preText = parts[0];
        streamState.buffer = parts[1] || '';

        if (preText) appendMainContent(stripChatMLTags(preText));
        streamState.inThinkingBlock = true;

        if (!streamState.thinkingDetails) {
            streamState.thinkingDetails = document.createElement('details');
            streamState.thinkingDetails.className = 'thinking-box';
            const summary = document.createElement('summary');
            summary.textContent = 'Thinking process';
            streamState.thinkingContent = document.createElement('div');
            streamState.thinkingContent.className = 'thinking-content';
            streamState.thinkingDetails.appendChild(summary);
            streamState.thinkingDetails.appendChild(streamState.thinkingContent);
            streamState.contentDiv.insertBefore(streamState.thinkingDetails, streamState.mainContentEl);
            streamState.thinkingDetails.open = true;
        }
    }

    // Exit thinking block
    if (streamState.inThinkingBlock && streamState.buffer.includes('</think>')) {
        const parts = streamState.buffer.split('</think>');
        const thinkText = parts[0];
        streamState.buffer = parts[1] || '';
        if (thinkText) streamState.thinkingContent.textContent += thinkText;
        streamState.inThinkingBlock = false;
        streamState.thinkingDetails.open = false;
    }

    // Flush buffer if no partial tag risk
    if (!streamState.buffer.includes('<')) {
        if (streamState.inThinkingBlock) {
            streamState.thinkingContent.textContent += streamState.buffer;
        } else {
            appendMainContent(stripChatMLTags(streamState.buffer));
        }
        streamState.buffer = '';
    }

    maybeScrollToBottom();
}

function finalizeStream(meta = {}) {
    if (!streamState.isStreaming) {
        if (stopBtn) stopBtn.classList.add('hidden');
        return;
    }

    if (streamState.buffer) {
        if (streamState.inThinkingBlock) {
            streamState.thinkingContent.textContent += streamState.buffer;
        } else {
            appendMainContent(stripChatMLTags(streamState.buffer));
        }
    }

    const elapsedMs = meta.elapsedMs || (streamStartTime ? Math.round(performance.now() - streamStartTime) : 0);
    const tokenCount = meta.tokenCount || streamTokenCount;

    const finalText = streamState.mainContentText || '';

    if (streamState.messageDiv && finalText.trim()) {
        const wrapper = streamState.messageDiv.querySelector('.message-content-wrapper');
        if (wrapper) {
            const meta = buildMessageMeta({ tokenCount, elapsedMs });
            const actions = buildAssistantActions(finalText, streamState.messageDiv);
            if (meta) wrapper.appendChild(meta);
            wrapper.appendChild(actions);
        }
    }
    // Don't poison history with an empty assistant turn — happens when the
    // model errors out, the user hits stop before any token arrives, or the
    // stream is cancelled by a newer request.
    if (currentMessages.length > 0 && finalText.trim()) {
        const data = { role: 'assistant', text: finalText };
        currentMessages.push(data);
        if (streamState.messageDiv) streamState.messageDiv._messageData = data;
        saveCurrentChat();
    } else if (currentMessages.length > 0 && streamState.messageDiv) {
        // No content was actually produced — drop the empty bubble so the
        // user doesn't see a phantom reply.
        streamState.messageDiv.remove();
    }

    streamState.isStreaming = false;
    streamState.messageDiv = null;
    streamState.contentDiv = null;
    streamState.mainContentEl = null;
    streamState.mainContentText = '';
    streamState.thinkingDetails = null;
    streamState.thinkingContent = null;
    streamState.buffer = '';
    streamState.inThinkingBlock = false;
    streamState.activeGenerationId = null;

    if (stopBtn) stopBtn.classList.add('hidden');

    // Restore the connected/ready badge once the queue/stream is settled.
    if (dc && dc.readyState === 'open') setStatus('connected', 'Ready');
}

// =============================================================================
// Message actions (copy / regenerate) and meta (timing)
// =============================================================================
function buildMessageMeta({ tokenCount, elapsedMs }) {
    if (!tokenCount && !elapsedMs) return null;
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    if (tokenCount) {
        const t = document.createElement('span');
        t.textContent = `${tokenCount} tokens`;
        meta.appendChild(t);
    }
    if (tokenCount && elapsedMs) {
        const dot = document.createElement('span');
        dot.className = 'message-meta-divider';
        meta.appendChild(dot);
    }
    if (elapsedMs) {
        const e = document.createElement('span');
        e.textContent = `${(elapsedMs / 1000).toFixed(2)}s`;
        meta.appendChild(e);
    }
    if (tokenCount && elapsedMs && elapsedMs > 0) {
        const dot = document.createElement('span');
        dot.className = 'message-meta-divider';
        meta.appendChild(dot);
        const rate = document.createElement('span');
        rate.textContent = `${(tokenCount / (elapsedMs / 1000)).toFixed(1)} tok/s`;
        meta.appendChild(rate);
    }
    return meta;
}

function buildAssistantActions(text, msgDiv) {
    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'message-action-btn';
    copyBtn.type = 'button';
    copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>';
    copyBtn.addEventListener('click', async () => {
        // Always grab the current bubble text — it may have changed since
        // the action bar was attached (e.g., after a regenerate).
        const live = (msgDiv && msgDiv._messageData && msgDiv._messageData.text) || text;
        try {
            await navigator.clipboard.writeText(live);
            copyBtn.classList.add('copied');
            copyBtn.querySelector('span').textContent = 'Copied';
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtn.querySelector('span').textContent = 'Copy';
            }, 1500);
        } catch (e) {
            showToast('Copy failed');
        }
    });

    const regenBtn = document.createElement('button');
    regenBtn.className = 'message-action-btn';
    regenBtn.type = 'button';
    regenBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15.5-6.4L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-15.5 6.4L3 16"></path><path d="M3 21v-5h5"></path></svg><span>Regenerate</span>';
    regenBtn.addEventListener('click', () => regenerateMessage(msgDiv));

    actions.appendChild(copyBtn);
    actions.appendChild(regenBtn);
    return actions;
}

function userPromptOf(msg) {
    return (msg && msg.text && msg.text.trim()) || (msg && msg.transcript) || '';
}

function regenerateMessage(assistantMsgDiv) {
    // Resolve the message the user wants to regenerate. If we don't have a
    // ref (legacy bubble created before per-message regenerate), fall back to
    // the most recent assistant message.
    let assistantData = assistantMsgDiv && assistantMsgDiv._messageData;
    if (!assistantData) {
        for (let i = currentMessages.length - 1; i >= 0; i--) {
            if (currentMessages[i].role === 'assistant') {
                assistantData = currentMessages[i];
                break;
            }
        }
    }
    if (!assistantData) {
        showToast('Nothing to regenerate');
        return;
    }
    const assistantIdx = currentMessages.indexOf(assistantData);
    if (assistantIdx < 1) {
        showToast('Nothing to regenerate');
        return;
    }

    // Walk back to the user turn that produced this assistant message.
    let userIdx = assistantIdx - 1;
    while (userIdx >= 0 && currentMessages[userIdx].role !== 'user') userIdx--;
    if (userIdx < 0) {
        showToast('Nothing to regenerate');
        return;
    }

    const userMsg = currentMessages[userIdx];
    const prompt = userPromptOf(userMsg);
    if (!prompt) {
        showToast('Nothing to regenerate');
        return;
    }

    // Count user turns from userIdx onward — these are the pairs the server
    // must roll back so its history matches our truncation.
    let dropPairs = 0;
    for (let i = userIdx; i < currentMessages.length; i++) {
        if (currentMessages[i].role === 'user') dropPairs++;
    }

    // Truncate local state to BEFORE the target user turn — sendTextOnly +
    // appendMessage will re-add it as part of the regenerated turn.
    const removed = currentMessages.splice(userIdx);
    revokeAudioUrls(removed);
    saveCurrentChat({ flush: true });

    // Remove DOM bubbles for the target user turn and everything after.
    const allMessages = Array.from(messagesContainer.querySelectorAll('.message'));
    let foundTarget = false;
    for (const m of allMessages) {
        if (foundTarget) { m.remove(); continue; }
        if (m._messageData === userMsg) {
            m.remove();
            foundTarget = true;
        }
    }

    appendMessage('user', prompt);
    sendTextOnly(prompt, { regenerate: true, dropPairs });
}

function revokeAudioUrls(messages) {
    if (!messages) return;
    for (const m of messages) {
        if (m && m.audioUrl && typeof m.audioUrl === 'string' && m.audioUrl.startsWith('blob:')) {
            try { URL.revokeObjectURL(m.audioUrl); } catch (e) { /* ignore */ }
        }
    }
}

// =============================================================================
// Negotiation
// =============================================================================
async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    if (pc.iceGatheringState !== 'complete') {
        await new Promise((resolve) => {
            const checkState = () => {
                if (pc.iceGatheringState === 'complete') {
                    pc.removeEventListener('icegatheringstatechange', checkState);
                    resolve();
                }
            };
            pc.addEventListener('icegatheringstatechange', checkState);
            setTimeout(resolve, 3000);
        });
    }

    const response = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
            sessionId,
        }),
    });

    if (!response.ok) throw new Error('Server returned error: ' + response.statusText);

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
}

// =============================================================================
// Composer
// =============================================================================
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
    stopBtn.addEventListener('click', stopGeneration);
}

function stopGeneration() {
    if (!streamState.isStreaming) return;
    streamState.stopRequested = true;
    if (dc && dc.readyState === 'open') {
        dc.send(JSON.stringify({ type: 'stop', generationId: streamState.currentGenerationId }));
    }
    finalizeStream();
}

removeFileBtn.addEventListener('click', () => {
    currentFile = null;
    audioInput.value = '';
    filePreview.classList.add('hidden');
});

uploadBtn.addEventListener('click', () => audioInput.click());

audioInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setCurrentFile(file);
});

function setCurrentFile(file) {
    currentFile = file;
    fileNameSpan.textContent = file.name;
    filePreview.classList.remove('hidden');
}

// =============================================================================
// Drag-and-drop
// =============================================================================
let dragCounter = 0;
function isAudioDrag(e) {
    if (!e.dataTransfer) return false;
    return Array.from(e.dataTransfer.items || []).some((it) => it.kind === 'file');
}

['dragenter', 'dragover'].forEach((evt) => {
    chatArea.addEventListener(evt, (e) => {
        if (!isAudioDrag(e)) return;
        e.preventDefault();
        if (evt === 'dragenter') dragCounter += 1;
        dropOverlay.classList.remove('hidden');
    });
});

chatArea.addEventListener('dragleave', () => {
    dragCounter = Math.max(0, dragCounter - 1);
    if (dragCounter === 0) dropOverlay.classList.add('hidden');
});

chatArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.classList.add('hidden');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    if (!file.type.startsWith('audio/') && !/\.(wav|mp3|m4a|ogg|flac|webm)$/i.test(file.name)) {
        showToast('Only audio files are supported');
        return;
    }
    setCurrentFile(file);
});

// =============================================================================
// Mic / Recording
// =============================================================================
micBtn.addEventListener('click', async () => {
    if (settings.micMode === 'ptt') return; // handled by mousedown/up
    if (streamState.isStreaming) finalizeStream();
    if (!isRecording) await startRecording();
    else await stopRecording();
});

// PTT: hold mic to record
function pttDown(e) {
    if (settings.micMode !== 'ptt') return;
    if (e.cancelable) e.preventDefault();
    if (pttHoldActive || isRecording) return;
    pttHoldActive = true;
    micBtn.classList.add('ptt-active');
    if (streamState.isStreaming) finalizeStream();
    startRecording({ ptt: true }).catch(() => { pttHoldActive = false; });
}
function pttUp() {
    if (settings.micMode !== 'ptt') return;
    if (!pttHoldActive) return;
    pttHoldActive = false;
    micBtn.classList.remove('ptt-active');
    if (isRecording) stopRecording();
}
micBtn.addEventListener('mousedown', pttDown);
micBtn.addEventListener('touchstart', pttDown, { passive: false });
window.addEventListener('mouseup', pttUp);
window.addEventListener('touchend', pttUp);

async function startRecording({ ptt = false } = {}) {
    try {
        if (pc) { pc.close(); pc = null; }

        await startWebRTC();

        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            },
            video: false,
        });

        localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

        // Tee the same stream into a local recorder so we can replay later.
        recordedChunks = [];
        try {
            const mime = pickRecorderMime();
            mediaRecorder = mime ? new MediaRecorder(localStream, { mimeType: mime }) : new MediaRecorder(localStream);
            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) recordedChunks.push(e.data);
            };
            mediaRecorder.onstop = () => {
                if (!recordedChunks.length || !pendingMicAudioMsg) return;
                const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                const url = URL.createObjectURL(blob);
                const contentDiv = pendingMicAudioMsg.querySelector('.content');
                if (contentDiv) {
                    const player = buildAudioPlayer(url);
                    const transcriptBox = contentDiv.querySelector('.transcript-box');
                    if (transcriptBox) contentDiv.insertBefore(player, transcriptBox);
                    else contentDiv.appendChild(player);
                }
                pendingMicAudioMsg._audioUrl = url;
                if (pendingMicAudioMsg._messageData) {
                    pendingMicAudioMsg._messageData.audioUrl = url;
                    saveCurrentChat();
                }
                pendingMicAudioMsg = null;
                recordedChunks = [];
            };
            mediaRecorder.start();
        } catch (err) {
            console.warn('MediaRecorder unavailable; recordings will not be playable.', err);
            mediaRecorder = null;
        }

        await negotiate();

        isRecording = true;
        micBtn.classList.add('recording');
        if (!ptt) {
            micBtn.setAttribute('aria-label', 'Stop recording');
            micBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="4" width="4" height="16" rx="1"></rect><rect x="14" y="4" width="4" height="16" rx="1"></rect></svg>';
        }

        showMeter();
        setStatus('connected', ptt ? 'Recording (hold)' : 'Listening…');

        if (vadController) vadController.stop();
        vadController = startVAD(localStream, {
            autoStop: !ptt,
            onSpeechStart: () => { if (!ptt) setStatus('connected', 'Recording'); },
            onSilence: (reason) => {
                console.log('VAD auto-stop:', reason);
                if (isRecording) stopRecording();
            },
            onLevel: updateMeter,
        });
    } catch (e) {
        console.error('Error starting recording:', e);
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
            showToast('Microphone access denied');
        } else if (location.hostname !== 'localhost' && location.protocol === 'http:') {
            showToast('Microphone requires HTTPS or localhost');
        } else {
            showToast('Could not access microphone');
        }
        setStatus('error', 'Mic unavailable');
        hideMeter();
        isRecording = false;
        micBtn.classList.remove('recording');
    }
}

async function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.setAttribute('aria-label', 'Start recording');
    micBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>';
    hideMeter();

    if (vadController) { vadController.stop(); vadController = null; }

    const prompt = textInput.value.trim();
    textInput.value = '';
    autoResizeTextarea();

    // Render the user message FIRST so pendingMicAudioMsg is set before
    // mediaRecorder fires onstop (otherwise we race and lose the blob).
    if (dc && dc.readyState === 'open') {
        if (streamState.isStreaming) finalizeStream();
        streamState.currentGenerationId += 1;
        dc.send(JSON.stringify({
            type: 'stop',
            text: prompt,
            generationId: streamState.currentGenerationId,
            maxTokens: settings.maxTokens,
            instruction: settings.instruction,
            effortMode: settings.effortMode,
        }));

        const userMsg = appendMessage('user', prompt || '', { transcript: '' });
        pendingMicAudioMsg = userMsg;
        pendingTranscriptMsg = userMsg;
        showThinkingMessage();
    }

    // Stop MediaRecorder, then defer stopping the underlying tracks until
    // onstop has fired — otherwise the recorder loses its final chunk in
    // some browsers and the audio shows up empty.
    const cleanupTracks = () => {
        if (localStream) {
            localStream.getTracks().forEach((track) => track.stop());
            localStream = null;
        }
    };

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        const prior = mediaRecorder.onstop;
        mediaRecorder.onstop = (ev) => {
            try { if (prior) prior.call(mediaRecorder, ev); }
            finally { cleanupTracks(); }
        };
        try { mediaRecorder.stop(); } catch (e) { cleanupTracks(); }
    } else {
        cleanupTracks();
    }
}

// =============================================================================
// Send / Upload
// =============================================================================
function sendMessage() {
    const text = textInput.value.trim();

    if (currentFile) {
        const blobUrl = URL.createObjectURL(currentFile);
        const userMsg = appendMessage(
            'user',
            text || '',
            { audioUrl: blobUrl, transcript: '' },
        );
        pendingTranscriptMsg = userMsg;
        requestUploadTranscript(currentFile, userMsg);
        uploadFile(currentFile, text);
        currentFile = null;
        filePreview.classList.add('hidden');
        audioInput.value = '';
        textInput.value = '';
        autoResizeTextarea();
        return;
    }

    if (!text) return;

    appendMessage('user', text);
    textInput.value = '';
    autoResizeTextarea();

    if (!isRecording) sendTextOnly(text);
}

async function sendTextOnly(text, { regenerate = false, dropPairs = 1 } = {}) {
    if (!pc) {
        try {
            await startWebRTC();
            await negotiate();
            await new Promise((resolve, reject) => {
                if (dc.readyState === 'open') return resolve();
                const cleanup = () => {
                    dc.removeEventListener('open', onOpen);
                    dc.removeEventListener('error', onError);
                    clearTimeout(timeoutId);
                };
                const onOpen = () => { cleanup(); resolve(); };
                const onError = (e) => { cleanup(); reject(new Error('DataChannel error: ' + e)); };
                const timeoutId = setTimeout(() => { cleanup(); reject(new Error('DataChannel timed out')); }, 10000);
                dc.addEventListener('open', onOpen);
                dc.addEventListener('error', onError);
            });
        } catch (e) {
            console.error('Connection failed:', e);
            setStatus('error', 'Connection failed');
            appendMessage('system', 'Error: Could not connect to server. Please refresh.');
            return;
        }
    }

    if (dc && dc.readyState === 'open') {
        if (streamState.isStreaming) finalizeStream();
        streamState.currentGenerationId += 1;
        dc.send(JSON.stringify({
            type: 'text_only',
            text,
            generationId: streamState.currentGenerationId,
            regenerate,
            dropPairs,
            maxTokens: settings.maxTokens,
            effortMode: settings.effortMode,
        }));
        showThinkingMessage();
    }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 60000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

async function requestUploadTranscript(file, userMsg) {
    let failureLabel = '(transcription unavailable)';
    try {
        const fd = new FormData();
        fd.append('audio', file);
        const res = await fetchWithTimeout('/transcribe', { method: 'POST', body: fd }, 45000);
        if (!res.ok) throw new Error('transcribe failed (' + res.status + ')');
        const data = await res.json();
        const text = (data && data.text) ? data.text : '';
        if (userMsg && userMsg._transcriptEl) {
            userMsg._transcriptEl.textContent = text || '(no speech detected)';
            userMsg._transcriptEl.classList.remove('placeholder');
        }
        if (userMsg && userMsg._messageData) {
            userMsg._messageData.transcript = text;
            saveCurrentChat();
        }
        return;
    } catch (e) {
        if (e.name === 'AbortError') failureLabel = '(transcription timed out)';
        console.warn('Transcript fetch failed:', e);
    }
    // Always settle the placeholder — never leave it spinning forever.
    if (userMsg && userMsg._transcriptEl) {
        userMsg._transcriptEl.textContent = failureLabel;
        userMsg._transcriptEl.classList.remove('placeholder');
    }
    if (userMsg && userMsg._messageData) {
        userMsg._messageData.transcript = '';
        saveCurrentChat();
    }
}

async function uploadFile(file, prompt) {
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('sessionId', sessionId);
    if (prompt) formData.append('text', prompt);
    formData.append('maxTokens', String(settings.maxTokens));
    formData.append('instruction', settings.instruction);
    formData.append('effortMode', settings.effortMode);

    const thinkingMsg = appendMessage('system', '');
    const thinkingIndicator = document.createElement('div');
    thinkingIndicator.className = 'thinking-indicator';
    thinkingIndicator.innerHTML = '<span></span><span></span><span></span>';
    thinkingMsg.querySelector('.content').appendChild(thinkingIndicator);

    try {
        const response = await fetchWithTimeout('/upload', { method: 'POST', body: formData }, 120000);
        thinkingMsg.remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage('system', data.text);
        } else {
            let errorText = 'Error uploading file.';
            try {
                const errorData = await response.text();
                try {
                    const jsonErr = JSON.parse(errorData);
                    if (jsonErr.text) errorText = jsonErr.text;
                    else if (jsonErr.message) errorText = jsonErr.message;
                } catch (e) {
                    if (errorData) errorText = `Error: ${errorData}`;
                }
            } catch (e) { /* ignore */ }
            appendMessage('system', errorText);
        }
    } catch (e) {
        console.error('Upload error:', e);
        thinkingMsg.remove();
        const msg = e.name === 'AbortError'
            ? 'The server took too long to respond. Try again or shorten the audio.'
            : 'Error uploading file.';
        appendMessage('system', msg);
    }
}

// =============================================================================
// Append message helpers
// =============================================================================
function showThinkingMessage() {
    currentThinkingMsg = appendMessage('system', '');
    const thinkingIndicator = document.createElement('div');
    thinkingIndicator.className = 'thinking-indicator';
    thinkingIndicator.innerHTML = '<span></span><span></span><span></span>';
    currentThinkingMsg.querySelector('.content').appendChild(thinkingIndicator);
}

function ensureEmptyStateRemoved() {
    const empty = messagesContainer.querySelector('.empty-state');
    if (empty) empty.remove();
}

// `transcript`: pass `null` to omit the dropdown entirely, `''` to render the
// "Transcribing…" placeholder, or any non-empty string to render the text.
function appendMessage(role, text, { silent = false, audioUrl = null, transcript = null } = {}) {
    ensureEmptyStateRemoved();

    // Storage uses 'assistant' (set by finalizeStream) but the existing CSS
    // and avatar logic was keyed on 'system'. Treat them as the same role.
    const isAssistant = role === 'system' || role === 'assistant';
    const cssRole = isAssistant ? 'system' : 'user';

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${cssRole}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.setAttribute('aria-hidden', 'true');
    avatarDiv.textContent = isAssistant ? 'AI' : 'U';

    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';

    if (isAssistant) {
        // Render markdown + parse <think> blocks first
        const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
        let match;
        let lastIndex = 0;
        const fragments = [];
        while ((match = thinkRegex.exec(text)) !== null) {
            const before = text.substring(lastIndex, match.index);
            if (before) fragments.push({ type: 'text', value: before });
            fragments.push({ type: 'think', value: match[1] });
            lastIndex = thinkRegex.lastIndex;
        }
        const remaining = text.substring(lastIndex);
        if (remaining) fragments.push({ type: 'text', value: remaining });
        if (fragments.length === 0 && text) fragments.push({ type: 'text', value: text });

        fragments.forEach((frag) => {
            if (frag.type === 'think') {
                const details = document.createElement('details');
                details.className = 'thinking-box';
                const summary = document.createElement('summary');
                summary.textContent = 'Thinking process';
                const body = document.createElement('div');
                body.className = 'thinking-content';
                body.textContent = frag.value;
                details.appendChild(summary);
                details.appendChild(body);
                contentDiv.appendChild(details);
            } else {
                const md = document.createElement('div');
                md.className = 'markdown-body';
                md.innerHTML = renderMarkdown(frag.value);
                contentDiv.appendChild(md);
            }
        });

        wrapper.appendChild(contentDiv);
        if (text) {
            wrapper.appendChild(buildAssistantActions(text, msgDiv));
        }
    } else {
        // Only render the text node when we actually have something to say —
        // avoids an empty "(Audio Message)" placeholder above the audio
        // player when the user sends voice without typing.
        if (text) {
            const textNode = document.createElement('div');
            textNode.className = 'message-text';
            textNode.textContent = text;
            contentDiv.appendChild(textNode);
        } else {
            contentDiv.classList.add('is-audio-only');
        }

        if (audioUrl) {
            const player = buildAudioPlayer(audioUrl);
            contentDiv.appendChild(player);
            msgDiv._audioUrl = audioUrl;
        }

        if (transcript !== null) {
            const details = document.createElement('details');
            details.className = 'transcript-box';
            const summary = document.createElement('summary');
            summary.textContent = 'Transcript';
            const body = document.createElement('div');
            body.className = 'transcript-content';
            if (transcript) {
                body.textContent = transcript;
            } else {
                body.textContent = 'Transcribing…';
                body.classList.add('placeholder');
            }
            details.appendChild(summary);
            details.appendChild(body);
            contentDiv.appendChild(details);
            msgDiv._transcriptEl = body;
        }

        wrapper.appendChild(contentDiv);
    }

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(wrapper);
    messagesContainer.appendChild(msgDiv);
    maybeScrollToBottom();

    if (!silent && (currentMessages.length > 0 || role === 'user')) {
        const data = { role, text };
        if (transcript !== null) data.transcript = transcript;
        if (audioUrl) data.audioUrl = audioUrl;
        currentMessages.push(data);
        msgDiv._messageData = data;
        saveCurrentChat();
    }

    return msgDiv;
}

// =============================================================================
// Keyboard shortcuts
// =============================================================================
document.addEventListener('keydown', (e) => {
    const meta = e.ctrlKey || e.metaKey;

    if (e.key === 'Escape') {
        if (!modalOverlay.classList.contains('hidden')) { closeModal(null); return; }
        if (!settingsOverlay.classList.contains('hidden')) { closeSettings(); return; }
        if (streamState.isStreaming) { stopGeneration(); return; }
    }

    if (meta && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        startNewChat();
        return;
    }

    if (meta && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleSidebar();
        return;
    }

    if (meta && e.key === ',') {
        e.preventDefault();
        if (settingsOverlay.classList.contains('hidden')) openSettings();
        else closeSettings();
        return;
    }

    if (meta && e.key === '/') {
        e.preventDefault();
        textInput.focus();
        return;
    }

    if (meta && e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
});

// =============================================================================
// Server stats: polling /healthz + /metrics every few seconds, rendering
// the small badge cluster, and feeding the modal viewer.
// =============================================================================
const serverStatsBtn = document.getElementById('server-stats-btn');
const serverStatsOverlay = document.getElementById('server-stats-overlay');
const serverStatsClose = document.getElementById('server-stats-close');
const serverStatsRefresh = document.getElementById('server-stats-refresh');
const serverStatsHealth = document.getElementById('server-stats-health');
const serverStatsMetrics = document.getElementById('server-stats-metrics');
const serverStatsUpdated = document.getElementById('server-stats-updated');

const statsModelPill = document.querySelector('.stats-model');
const statsQueuePill = document.querySelector('.stats-queue');
const statsQueueLabel = document.getElementById('stats-queue-label');
const statsQueueBars = document.querySelectorAll('.stats-queue .stats-bars i');
const statsSessionsLabel = document.getElementById('stats-sessions-label');
const modelNamePill = document.getElementById('model-name-pill');
const modelNameValue = document.getElementById('model-name-value');

let lastHealth = null;
let lastMetrics = null;
let statsPollTimer = null;

function renderStatsBadges() {
    // Model pill
    const modelOk = !!(lastHealth && lastHealth.model_loaded);
    if (statsModelPill) {
        statsModelPill.dataset.state = lastHealth ? (modelOk ? 'ok' : 'error') : 'unknown';
        const label = statsModelPill.querySelector('.stats-label');
        if (label) label.textContent = modelOk ? 'model' : (lastHealth ? 'no model' : 'offline');
    }

    // Model name pill (top-left of header). Name comes from MODEL_PATH on
    // the server and is always shown when reachable, even if the weights
    // failed to load — load state is reflected by the pill dot color.
    if (modelNamePill && modelNameValue) {
        const name = lastHealth && lastHealth.model_name;
        if (!lastHealth) {
            modelNamePill.dataset.state = 'offline';
            modelNameValue.textContent = 'server unreachable';
            modelNamePill.title = 'Server unreachable';
        } else {
            modelNamePill.dataset.state = modelOk ? 'ok' : 'loading';
            modelNameValue.textContent = name || 'unknown';
            modelNamePill.title = modelOk
                ? `Active model: ${name || 'unknown'}`
                : `Model defined but not loaded: ${name || 'unknown'}`;
        }
    }

    // Queue bar visualization
    if (statsQueuePill && lastMetrics && lastMetrics.queue) {
        const q = lastMetrics.queue;
        const depth = q.depth || 0;
        const max = q.max_depth || 5;
        const filled = Math.min(statsQueueBars.length, Math.round((depth / max) * statsQueueBars.length));
        statsQueueBars.forEach((bar, i) => bar.classList.toggle('lit', i < filled));
        statsQueueLabel.textContent = `Q ${depth}/${max}`;
        const ratio = depth / max;
        statsQueuePill.dataset.state = ratio >= 1 ? 'error' : ratio >= 0.6 ? 'busy' : 'ok';
    } else if (statsQueuePill) {
        statsQueuePill.dataset.state = 'unknown';
        statsQueueLabel.textContent = 'queue';
    }

    // Active connections (live peer connections, not historical sessions).
    if (statsSessionsLabel && lastMetrics) {
        statsSessionsLabel.textContent = String(lastMetrics.active_pcs ?? 0);
    }
}

function renderStatsModalContent() {
    if (!serverStatsHealth || !serverStatsMetrics) return;
    serverStatsHealth.textContent = lastHealth ? JSON.stringify(lastHealth, null, 2) : '(unreachable)';
    serverStatsMetrics.textContent = lastMetrics ? JSON.stringify(lastMetrics, null, 2) : '(unreachable)';
    if (serverStatsUpdated) {
        serverStatsUpdated.textContent = lastHealth || lastMetrics ? `Updated ${new Date().toLocaleTimeString()}` : '';
    }
}

async function pollServerStats() {
    try {
        const [hRes, mRes] = await Promise.all([
            fetch('/healthz').catch(() => null),
            fetch('/metrics').catch(() => null),
        ]);
        lastHealth = hRes && hRes.ok ? await hRes.json() : null;
        lastMetrics = mRes && mRes.ok ? await mRes.json() : null;
    } catch (e) {
        lastHealth = null;
        lastMetrics = null;
    }
    renderStatsBadges();
    if (serverStatsOverlay && !serverStatsOverlay.classList.contains('hidden')) {
        renderStatsModalContent();
    }
}

function startStatsPolling() {
    if (statsPollTimer) return;
    pollServerStats();
    statsPollTimer = setInterval(() => {
        if (document.visibilityState === 'visible') pollServerStats();
    }, 5000);
}

if (serverStatsBtn) {
    serverStatsBtn.addEventListener('click', () => {
        renderStatsModalContent();
        serverStatsOverlay.classList.remove('hidden');
        pollServerStats();
    });
}

// ---- Model config modal -----------------------------------------------------
const modelConfigOverlay = document.getElementById('model-config-overlay');
const modelConfigClose = document.getElementById('model-config-close');
const modelConfigCopy = document.getElementById('model-config-copy');
const modelConfigBody = document.getElementById('model-config-body');
const modelConfigPath = document.getElementById('model-config-path');

async function openModelConfig() {
    if (!modelConfigOverlay) return;
    modelConfigOverlay.classList.remove('hidden');
    modelConfigBody.textContent = 'Loading…';
    modelConfigPath.textContent = '';
    try {
        const res = await fetchWithTimeout('/model-config', {}, 10000);
        const data = await res.json();
        if (res.ok) {
            modelConfigPath.textContent = data.model_path || '';
            modelConfigBody.textContent = JSON.stringify(data.config, null, 2);
        } else {
            modelConfigPath.textContent = data.model_path || '';
            modelConfigBody.textContent = data.error || `HTTP ${res.status}`;
        }
    } catch (e) {
        modelConfigBody.textContent = e.name === 'AbortError'
            ? 'Request timed out.'
            : 'Could not reach the server.';
    }
}

if (modelNamePill) {
    modelNamePill.addEventListener('click', openModelConfig);
}
if (modelConfigClose) {
    modelConfigClose.addEventListener('click', () => modelConfigOverlay.classList.add('hidden'));
}
if (modelConfigOverlay) {
    modelConfigOverlay.addEventListener('click', (e) => {
        if (e.target === modelConfigOverlay) modelConfigOverlay.classList.add('hidden');
    });
}
if (modelConfigCopy) {
    modelConfigCopy.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(modelConfigBody.textContent || '');
            modelConfigCopy.textContent = 'Copied';
            setTimeout(() => { modelConfigCopy.textContent = 'Copy'; }, 1500);
        } catch (e) {
            showToast('Copy failed');
        }
    });
}
if (serverStatsClose) {
    serverStatsClose.addEventListener('click', () => serverStatsOverlay.classList.add('hidden'));
}
if (serverStatsRefresh) {
    serverStatsRefresh.addEventListener('click', pollServerStats);
}
if (serverStatsOverlay) {
    serverStatsOverlay.addEventListener('click', (e) => {
        if (e.target === serverStatsOverlay) serverStatsOverlay.classList.add('hidden');
    });
}

// =============================================================================
// Init + cleanup
// =============================================================================
window.addEventListener('beforeunload', () => {
    saveCurrentChat({ flush: true });
    revokeAudioUrls(currentMessages);
    const history = loadChatHistory();
    history.chats.forEach((c) => revokeAudioUrls(c.messages));
});

renderEmptyState();
renderChatHistory();
startStatsPolling();
