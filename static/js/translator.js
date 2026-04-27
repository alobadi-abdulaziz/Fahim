const chat     = document.getElementById('chat');
const inp      = document.getElementById('inp');
const sendBtn  = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const empty    = document.getElementById('empty');

inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
});

inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

sendBtn.addEventListener('click', send);

function now() {
  return new Date().toLocaleTimeString('ar', { hour: '2-digit', minute: '2-digit' });
}

function hideEmpty() {
  if (empty) empty.style.display = 'none';
}

function addMsg(role, html) {
  hideEmpty();
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="bubble">${html}</div><div class="meta">${now()}</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function addTyping() {
  hideEmpty();
  const el = document.createElement('div');
  el.className = 'msg bot typing';
  el.innerHTML = `<div class="bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function buildBotHTML(matched, notFound, videoUrl) {
  const matchTags = matched.map(w => `<span class="tag">${w}</span>`).join('');
  const missTags  = notFound.length
    ? `<div style="margin-top:8px;font-size:12px;color:var(--danger);">لم يُوجد لها مقابل:</div>
       <div class="tags">${notFound.map(w => `<span class="tag miss">${w}</span>`).join('')}</div>`
    : '';
  const video = videoUrl
    ? `<div class="video-wrap">
         <video src="${videoUrl}?t=${Date.now()}" autoplay controls playsinline></video>
       </div>`
    : '';

  return `<div style="font-size:13px;color:var(--text-sub);margin-bottom:6px;">الكلمات المترجمة</div>
          <div class="tags">${matchTags}</div>
          ${missTags}
          ${video}`;
}

async function send() {
  const text = inp.value.trim();
  if (!text) return;

  inp.value = '';
  inp.style.height = 'auto';
  sendBtn.disabled = true;

  addMsg('user', text);
  const typing = addTyping();

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    typing.remove();

    if (!res.ok) {
      addMsg('bot', `⚠️ ${data.error || 'حدث خطأ'}`);
    } else {
      addMsg('bot', buildBotHTML(data.matched, data.not_found || [], data.video_url));
    }
  } catch {
    typing.remove();
    addMsg('bot', '⚠️ تعذر الاتصال بالسيرفر');
  }

  sendBtn.disabled = false;
  inp.focus();
}

// ── Voice ──
let recording = false;

voiceBtn.addEventListener('click', async () => {
  if (recording) return;
  recording = true;
  voiceBtn.classList.add('recording');
  voiceBtn.textContent = '⏹';

  try {
    const res  = await fetch('/voice', { method: 'POST' });
    const data = await res.json();

    if (data.text) {
      inp.value = data.text;
      inp.style.height = 'auto';
      inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
      inp.focus();
    } else {
      addMsg('bot', '⚠️ لم يُتعرف على الصوت، حاول مرة أخرى');
    }
  } catch {
    addMsg('bot', '⚠️ خطأ في التسجيل');
  }

  recording = false;
  voiceBtn.classList.remove('recording');
  voiceBtn.textContent = '🎤';
});
