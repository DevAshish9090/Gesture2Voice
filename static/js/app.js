const socket = io();
let history = [];

socket.on('connect', () => {
    document.getElementById('statusDot').classList.add('active');
    document.getElementById('statusText').textContent = 'Camera active';
    document.getElementById('cameraOverlay').style.display = 'none';
});

socket.on('prediction_update', (data) => {
    const wordEl = document.getElementById('detectedWord');
    const confBar = document.getElementById('confBar');
    const confValue = document.getElementById('confValue');

    if (data.no_hand) {
        wordEl.textContent = '—';
        confBar.style.width = '0%';
        confValue.textContent = '0%';
        return;
    }

    wordEl.textContent = data.word || '—';
    confBar.style.width = data.confidence + '%';
    confValue.textContent = data.confidence + '%';

    if (data.confidence > 70) {
        confBar.style.background = '#10b981';
    } else if (data.confidence > 40) {
        confBar.style.background = '#f59e0b';
    } else {
        confBar.style.background = '#2563eb';
    }
});

socket.on('word_added', (data) => {
    const words = data.sentence.replace('.', '').split(' ');
    updateSentenceDisplay(data.sentence);
    updateWordChips(words);
});

socket.on('sentence_update', (data) => {
    updateSentenceDisplay('');
    updateWordChips([]);
});

socket.on('emergency_triggered', () => {
    if (document.getElementById('emergencyBtn').classList.contains('active')) {
        showEmergency();
    }
});

function updateSentenceDisplay(text) {
    const box = document.getElementById('sentenceBox');
    if (!text || text === '.') {
        box.innerHTML = '<span class="sentence-placeholder">Gestures will appear here...</span>';
    } else {
        box.textContent = text;
    }
}

function updateWordChips(words) {
    const chips = document.getElementById('wordChips');
    chips.innerHTML = '';
    words.forEach((word, i) => {
        if (!word) return;
        const chip = document.createElement('div');
        chip.className = 'chip' + (i === words.length - 1 ? ' new' : '');
        chip.textContent = word;
        chips.appendChild(chip);
    });
}

function speakSentence() {
    socket.emit('speak', {});
}

function resetSentence() {
    socket.emit('reset');
}

function setMode(mode) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + mode).classList.add('active');
    document.getElementById('modeBadge').textContent = mode.toUpperCase() + ' MODE';
}

function setVoice() {
    const voice = document.getElementById('voiceSelect').value;
    socket.emit('set_voice', { voice });
}

function toggleEmergency() {
    const btn = document.getElementById('emergencyBtn');
    const status = document.getElementById('emergencyStatus');
    btn.classList.toggle('active');
    status.textContent = btn.classList.contains('active') ? 'ACTIVE' : 'INACTIVE';
}

function showEmergency() {
    document.getElementById('emergencyOverlay').classList.add('show');
    socket.emit('emergency_speak');
}

function dismissEmergency() {
    document.getElementById('emergencyOverlay').classList.remove('show');
}

function updateHistory(sentence) {
    if (!sentence) return;
    history.unshift(sentence);
    if (history.length > 5) history.pop();
    const list = document.getElementById('historyList');
    list.innerHTML = '';
    history.forEach(item => {
        const el = document.createElement('div');
        el.className = 'history-item';
        el.textContent = item;
        el.onclick = () => socket.emit('speak_text', { text: item });
        list.appendChild(el);
    });
}