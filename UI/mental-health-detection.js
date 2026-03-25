const API_BASE = "http://127.0.0.1:8000";

let curPage = 0;
const progs = [0, 25, 50, 75, 100];

const modalityScores = {
  text: null,
  audio: null,
  video: null,
};

const modalityRaw = {
  text: null,
  audio: null,
  video: null,
};

const fusedScores = {
  stress: 0,
  anxiety: 0,
  depression: 0,
};

const STEP_DELAY_MS = 5500;
const FINAL_WAIT_MS = 20000;
const pendingPredictions = {
  text: null,
  audio: null,
  video: null,
};

const apiErrors = {
  text: null,
  audio: null,
  video: null,
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

function speakPrompt(message) {
  const synth = window.speechSynthesis;
  if (!synth || !message) return;
  try {
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;
    synth.speak(utterance);
  } catch (err) {
    console.warn("Speech prompt failed:", err);
  }
}

function goToPage(n) {
  document.getElementById("page-" + curPage).classList.remove("active");
  curPage = n;
  document.getElementById("page-" + n).classList.add("active");
  document.getElementById("prog").style.width = progs[n] + "%";
  scrollTo(0, 0);
  if (n === 4) {
    computeFusedScores();
    renderExplainability();
    setTimeout(animResults, 350);
    setTimeout(showChat, 1200);
  }
}

function onTextInput(el) {
  const l = el.value.length;
  const cc = document.getElementById("char-cnt");
  cc.textContent = l + " / 50";
  cc.className = "char-counter" + (l >= 50 ? " ok" : "");
  document.getElementById("btn-text-next").disabled = l < 50;
}

function addChip(el, label) {
  el.classList.toggle("on");
  const ta = document.getElementById("text-input");
  if (el.classList.contains("on")) {
    ta.value += (ta.value ? ", " : "") + label.toLowerCase();
    onTextInput(ta);
  }
}

function showOverlay(id, show) {
  const el = document.getElementById(id);
  if (!el) return;
  if (show) el.classList.add("show");
  else el.classList.remove("show");
}

function parseApiScores(scores) {
  return {
    stress: Number(scores.stress || 0),
    anxiety: Number(scores.anxiety || 0),
    depression: Number(scores.depression || 0),
  };
}

async function analyzeText() {
  const text = document.getElementById("text-input").value.trim();
  if (text.length < 10) {
    alert("Please enter at least 10 characters.");
    return;
  }

  showOverlay("text-overlay", true);
  document.getElementById("btn-text-next").disabled = true;

  pendingPredictions.text = (async () => {
    const res = await fetch(`${API_BASE}/predict/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Text prediction failed");
    }
    const data = await res.json();
    modalityScores.text = parseApiScores(data.scores);
    modalityRaw.text = data.raw || null;
    apiErrors.text = null;
  })().catch((err) => {
    apiErrors.text = err.message || "Text prediction failed.";
    console.error("Text prediction failed:", err);
  });

  await sleep(STEP_DELAY_MS);
  showOverlay("text-overlay", false);
  goToPage(2);
}

function onAudioFileSelected(input) {
  const btn = document.getElementById("btn-audio-next");
  if (input.files && input.files.length > 0) {
    btn.disabled = false;
    recordedAudioLevel = 1;
    document.getElementById("status-txt").textContent = `Selected: ${input.files[0].name}`;
  }
}

let recOn = false;
let recInterval;
let recSecs = 0;
let mediaRecorder = null;
let micStream = null;
let recordedChunks = [];
let recordedAudioBlob = null;
let recordedAudioExt = "wav";
let audioContext = null;
let audioSourceNode = null;
let audioProcessorNode = null;
let pcmChunks = [];
let recordedSampleRate = 44100;
let recordedAudioLevel = 0;
const wf = document.getElementById("waveform");
for (let i = 0; i < 40; i++) {
  const b = document.createElement("div");
  b.className = "wb";
  b.style.animationDelay = i * 0.035 + "s";
  wf.appendChild(b);
}

async function toggleRec() {
  recOn = !recOn;
  const btn = document.getElementById("rec-btn");
  const zone = document.getElementById("rec-zone");
  const tmr = document.getElementById("timer");
  const st = document.getElementById("rec-status");
  const stxt = document.getElementById("status-txt");

  if (recOn) {
    recordedAudioBlob = null;
    recordedAudioExt = "wav";
    recordedAudioLevel = 0;
    recordedChunks = [];
    pcmChunks = [];

    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      recordedSampleRate = audioContext.sampleRate || 44100;
      audioSourceNode = audioContext.createMediaStreamSource(micStream);
      audioProcessorNode = audioContext.createScriptProcessor(4096, 1, 1);
      audioProcessorNode.onaudioprocess = (event) => {
        const channelData = event.inputBuffer.getChannelData(0);
        pcmChunks.push(new Float32Array(channelData));
      };
      audioSourceNode.connect(audioProcessorNode);
      audioProcessorNode.connect(audioContext.destination);
    } catch (err) {
      recOn = false;
      alert("Microphone access failed. Please allow mic permission.");
      console.error(err);
      return;
    }

    btn.classList.add("recording");
    btn.textContent = "STOP";
    zone.classList.add("recording");
    tmr.classList.add("live");
    st.classList.add("live");
    stxt.textContent = "Recording in progress...";
    wf.classList.add("live");
    recSecs = 0;
    recInterval = setInterval(() => {
      recSecs++;
      tmr.textContent =
        String(Math.floor(recSecs / 60)).padStart(2, "0") +
        ":" +
        String(recSecs % 60).padStart(2, "0");
      wf.querySelectorAll(".wb").forEach((bar) => {
        bar.style.height = Math.random() * 30 + 4 + "px";
      });
      if (recSecs >= 60) toggleRec();
    }, 1000);
  } else {
    recordedAudioBlob = encodeWavFromChunks(pcmChunks, recordedSampleRate);
    recordedAudioExt = "wav";
    recordedAudioLevel = estimateAudioLevel(pcmChunks);

    if (audioProcessorNode) {
      audioProcessorNode.disconnect();
      audioProcessorNode.onaudioprocess = null;
      audioProcessorNode = null;
    }
    if (audioSourceNode) {
      audioSourceNode.disconnect();
      audioSourceNode = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    mediaRecorder = null;

    btn.classList.remove("recording");
    btn.textContent = "MIC";
    zone.classList.remove("recording");
    tmr.classList.remove("live");
    st.classList.remove("live");
    stxt.textContent = `Complete - ${recSecs}s captured`;
    wf.classList.remove("live");
    wf.querySelectorAll(".wb").forEach((bar) => {
      bar.style.height = "4px";
    });
    clearInterval(recInterval);
    if (recSecs >= 5 && recordedAudioLevel >= 0.008) {
      document.getElementById("btn-audio-next").disabled = false;
      stxt.textContent = `Complete - ${recSecs}s captured (ready to analyze)`;
    } else if (recSecs >= 5) {
      document.getElementById("btn-audio-next").disabled = true;
      stxt.textContent = "No clear voice detected. Please record again and speak clearly.";
    }
  }
}

function encodeWavFromChunks(chunks, sampleRate) {
  if (!chunks || chunks.length === 0) return null;

  let totalLength = 0;
  for (const chunk of chunks) totalLength += chunk.length;

  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const wavBuffer = new ArrayBuffer(44 + merged.length * 2);
  const view = new DataView(wavBuffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + merged.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, merged.length * 2, true);

  let index = 44;
  for (let i = 0; i < merged.length; i++) {
    const sample = Math.max(-1, Math.min(1, merged[i]));
    view.setInt16(index, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    index += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function estimateAudioLevel(chunks) {
  if (!chunks || chunks.length === 0) return 0;
  let total = 0;
  let count = 0;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      total += Math.abs(chunk[i]);
      count += 1;
    }
  }
  return count ? total / count : 0;
}

async function analyzeAudio() {
  const input = document.getElementById("audio-file");
  const hasUpload = input && input.files && input.files.length > 0;
  const hasRecording = recordedAudioBlob && recordedAudioBlob.size > 0;

  if (!hasRecording && !hasUpload) {
    alert("Please record audio first.");
    return;
  }

  showOverlay("audio-overlay", true);
  document.getElementById("btn-audio-next").disabled = true;

  pendingPredictions.audio = (async () => {
    const form = new FormData();
    if (hasRecording) {
      form.append("file", recordedAudioBlob, `recorded_audio.${recordedAudioExt}`);
    } else {
      form.append("file", input.files[0]);
    }
    const res = await fetch(`${API_BASE}/predict/audio`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Audio prediction failed");
    }
    const data = await res.json();
    modalityRaw.audio = data.raw || null;
    modalityScores.audio = parseApiScores(data.scores);
    apiErrors.audio = null;
  })().catch((err) => {
    apiErrors.audio = err.message || "Audio prediction failed.";
    console.error("Audio prediction failed:", err);
  });

  await sleep(STEP_DELAY_MS);
  showOverlay("audio-overlay", false);
  goToPage(3);
}

let camStream = null;

function startCam() {
  const go = () => {
    document.getElementById("cam-ph").style.display = "none";
    ["face-box", "scan-beam", "cc-tl", "cc-tr", "cc-bl", "cc-br"].forEach((id) =>
      document.getElementById(id).classList.add("on")
    );
    document.getElementById("cam-start-btn").style.display = "none";
    document.getElementById("cam-stop-btn").style.display = "flex";
    document.getElementById("btn-video-next").disabled = false;
    speakPrompt("Please place your face in front of the camera.");
  };

  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((s) => {
      camStream = s;
      const v = document.getElementById("cam-feed");
      v.srcObject = s;
      v.classList.add("on");
      go();
    })
    .catch(() => go());
}

function stopCam() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  document.getElementById("cam-feed").classList.remove("on");
  document.getElementById("cam-ph").style.display = "";
  document.getElementById("cam-ph").querySelector(".cam-placeholder-text").textContent =
    "Camera stopped";
  ["face-box", "scan-beam", "cc-tl", "cc-tr", "cc-bl", "cc-br"].forEach((id) =>
    document.getElementById(id).classList.remove("on")
  );
  document.getElementById("cam-stop-btn").style.display = "none";
  document.getElementById("cam-start-btn").style.display = "flex";
  document.getElementById("cam-start-btn").innerHTML = "Restart Camera";
}

async function analyzeVideo() {
  const frameBlob = captureCurrentVideoFrame();
  if (!frameBlob) {
    speakPrompt("Please start the camera and place your face in front of the camera.");
    alert("Please start the camera and wait for the live feed before analyzing video.");
    return;
  }

  showOverlay("video-overlay", true);
  document.getElementById("btn-video-next").disabled = true;

  pendingPredictions.video = (async () => {
    const form = new FormData();
    form.append("file", frameBlob, "camera-frame.jpg");
    const res = await fetch(`${API_BASE}/predict/video`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Video prediction failed");
    }
    const data = await res.json();
    modalityRaw.video = data.raw || null;
    modalityScores.video = data.raw && data.raw.available === false ? null : parseApiScores(data.scores);
    if (data.raw && data.raw.available === false) {
      speakPrompt("No face detected. Please place your face in front of the camera and try again.");
    }
    apiErrors.video = null;
  })().catch((err) => {
    apiErrors.video = err.message || "Video prediction failed.";
    console.error("Video prediction failed:", err);
  });

  stopCam();

  const waits = [];
  if (pendingPredictions.text) waits.push(withTimeout(pendingPredictions.text, FINAL_WAIT_MS));
  if (pendingPredictions.audio) waits.push(withTimeout(pendingPredictions.audio, FINAL_WAIT_MS));
  if (pendingPredictions.video) waits.push(withTimeout(pendingPredictions.video, FINAL_WAIT_MS));

  if (waits.length > 0) {
    await Promise.allSettled(waits);
  } else {
    await sleep(1500);
  }

  showOverlay("video-overlay", false);
  goToPage(4);
}

function computeFusedScores() {
  const valid = [modalityScores.text, modalityScores.audio, modalityScores.video].filter(Boolean);
  if (valid.length === 0) {
    fusedScores.stress = 0;
    fusedScores.anxiety = 0;
    fusedScores.depression = 0;
    return;
  }

  const sum = valid.reduce(
    (acc, m) => {
      acc.stress += m.stress || 0;
      acc.anxiety += m.anxiety || 0;
      acc.depression += m.depression || 0;
      return acc;
    },
    { stress: 0, anxiety: 0, depression: 0 }
  );

  fusedScores.stress = Math.round(sum.stress / valid.length);
  fusedScores.anxiety = Math.round(sum.anxiety / valid.length);
  fusedScores.depression = Math.round(sum.depression / valid.length);
}

function explainSeverity(score, label) {
  if (score >= 70) return `${label} was one of the strongest signals in this modality.`;
  if (score >= 45) return `${label} appeared at a moderate level in this modality.`;
  return `${label} was present but not dominant in this modality.`;
}

function textExplanationItems() {
  const raw = modalityRaw.text;
  const scores = modalityScores.text;
  if (!raw || !scores) {
    return ["Text explanation is unavailable for this run."];
  }

  const items = [];
  if (raw.depression?.level) {
    items.push(
      `Depression-related language was classified as ${raw.depression.level}, which pushed the text depression score upward.`
    );
  }
  if (raw.stress?.source === "proxy_from_text_keywords") {
    items.push(
      `Stress percentage was influenced by stress-related keywords and phrasing in the written response.`
    );
  }
  if (raw.anxiety?.source === "proxy_from_text_keywords") {
    items.push(
      `Anxiety percentage was influenced by worry, fear, or nervousness-related wording in the text.`
    );
  }
  items.push(explainSeverity(scores.depression, "Depression"));
  return items.slice(0, 3);
}

function audioExplanationItems() {
  const scores = modalityScores.audio;
  if (!scores) {
    return ["Audio explanation is unavailable for this run."];
  }

  const ordered = [
    ["stress", scores.stress],
    ["depression", scores.depression],
    ["anxiety", scores.anxiety],
  ].sort((a, b) => b[1] - a[1]);

  return [
    `${capitalize(ordered[0][0])} had the strongest contribution from the vocal model at ${ordered[0][1].toFixed(1)}%.`,
    `${capitalize(ordered[1][0])} remained present in the voice signal, but below the strongest audio pattern.`,
    `Audio percentages come from learned vocal markers in the recording rather than text meaning or facial cues.`,
  ];
}

function videoExplanationItems() {
  const raw = modalityRaw.video;
  const scores = modalityScores.video;
  if (!raw || !scores || raw.available === false) {
    if (raw?.reason) {
      return [raw.reason];
    }
    return ["Video explanation is unavailable for this run."];
  }

  const items = [];
  if (raw.dominant_emotion) {
    items.push(
      `The facial model saw ${raw.dominant_emotion} as the dominant visible emotion in the captured frame.`
    );
  }
  if (raw.emotions?.sad !== undefined) {
    items.push(`Depression score was derived mainly from the model's sad emotion probability of ${Number(raw.emotions.sad).toFixed(1)}%.`);
  }
  if (raw.emotions?.fear !== undefined) {
    items.push(`Anxiety score was derived mainly from the model's fear emotion probability of ${Number(raw.emotions.fear).toFixed(1)}%.`);
  }
  if (raw.emotions?.angry !== undefined) {
    items.push(`Stress score was derived mainly from the model's angry emotion probability of ${Number(raw.emotions.angry).toFixed(1)}%.`);
  }
  return items.slice(0, 3);
}

function capitalize(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : "";
}

function renderExplainability() {
  renderResultsAlert();
  renderExplainabilitySection("xai-text-list", textExplanationItems());
  renderExplainabilitySection("xai-audio-list", audioExplanationItems());
  renderExplainabilitySection("xai-video-list", videoExplanationItems());
}

function renderResultsAlert() {
  const el = document.getElementById("results-alert");
  if (!el) return;
  const activeErrors = Object.entries(apiErrors)
    .filter(([, value]) => value)
    .map(([key]) => key);

  if (activeErrors.length === 0) {
    el.classList.remove("show");
    el.textContent = "";
    return;
  }

  el.classList.add("show");
  el.textContent = `Live backend analysis is unavailable for: ${activeErrors.join(", ")}. Showing zero values instead of fallback demo scores.`;
}

function renderExplainabilitySection(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function captureCurrentVideoFrame() {
  const video = document.getElementById("cam-feed");
  if (!video || !video.srcObject || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  const [header, base64] = dataUrl.split(",");
  if (!base64 || !header.includes("image/jpeg")) return null;

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: "image/jpeg" });
}

function animateMetric(arcId, valId, barId, target) {
  const arc = document.getElementById(arcId);
  const val = document.getElementById(valId);
  const bar = document.getElementById(barId);

  bar.dataset.w = String(target);
  bar.style.width = target + "%";

  setTimeout(() => {
    arc.style.strokeDashoffset = String(176 - (176 * target) / 100);
  }, 300);

  let c = 0;
  const step = Math.max(1, Math.floor(target / 25));
  const iv = setInterval(() => {
    c = Math.min(c + step, target);
    val.textContent = String(c);
    if (c >= target) clearInterval(iv);
  }, 30);
}

function animResults() {
  animateMetric("arc-stress", "val-stress", "bar-stress", fusedScores.stress);
  animateMetric("arc-anxiety", "val-anxiety", "bar-anxiety", fusedScores.anxiety);
  animateMetric("arc-depress", "val-depress", "bar-depress", fusedScores.depression);
}

let chatOpen = false;
function showChat() {
  document.getElementById("chat-fab").classList.add("show");
  setTimeout(toggleChat, 800);
}

function toggleChat() {
  chatOpen = !chatOpen;
  const p = document.getElementById("chat-panel");
  const n = document.getElementById("chat-notif");
  if (chatOpen) {
    p.classList.add("open");
    n.style.display = "none";
  } else {
    p.classList.remove("open");
  }
}

function severityLabel(score) {
  if (score >= 70) return "high";
  if (score >= 40) return "moderate";
  return "low";
}

function highestMetric() {
  const metrics = [
    ["stress", fusedScores.stress],
    ["anxiety", fusedScores.anxiety],
    ["depression", fusedScores.depression],
  ].sort((a, b) => b[1] - a[1]);
  return metrics[0];
}

function assistantReply(action) {
  const [topMetric, topScore] = highestMetric();
  const topSeverity = severityLabel(topScore);

  if (action === "explain") {
    return `Your highest indicator is ${topMetric} at ${topScore}%. Stress is ${fusedScores.stress}%, anxiety is ${fusedScores.anxiety}%, and depression is ${fusedScores.depression}%. These values come from the fused multimodal screening pipeline.`;
  }
  if (action === "stress") {
    return `Your stress indicator is ${fusedScores.stress}% (${severityLabel(fusedScores.stress)}). Short breaks, better sleep timing, hydration, and paced breathing may help reduce overload.`;
  }
  if (action === "anxiety") {
    return `Your anxiety indicator is ${fusedScores.anxiety}% (${severityLabel(fusedScores.anxiety)}). Grounding exercises, slower breathing, and reducing overstimulation can help manage anxious symptoms.`;
  }
  if (action === "depression") {
    return `Your depression indicator is ${fusedScores.depression}% (${severityLabel(fusedScores.depression)}). Maintaining routine, sunlight exposure, rest, and social support may be useful first steps.`;
  }
  if (action === "help") {
    return `If your strongest concern is ${topMetric} and it feels persistent or affects daily life, consider speaking with a counselor, psychologist, or licensed mental health professional.`;
  }
  if (action === "diagnosis") {
    return `No. This system provides screening indicators only. Your current highest signal is ${topMetric} at a ${topSeverity} level, but it is not a clinical diagnosis.`;
  }
  return "Choose one of the quick actions to explore your result.";
}

function runAssistantAction(action) {
  const labels = {
    explain: "Explain My Scores",
    stress: "Stress Tips",
    anxiety: "Anxiety Tips",
    depression: "Depression Tips",
    help: "When To Seek Help",
    diagnosis: "Is This A Diagnosis?",
  };

  const txt = labels[action] || "Assistant Action";
  const msgs = document.getElementById("chat-msgs");
  const u = document.createElement("div");
  u.className = "msg user";
  u.innerHTML = `<div class="msg-av">U</div><div class="msg-bub">${txt}</div>`;
  msgs.appendChild(u);

  const typing = document.createElement("div");
  typing.className = "msg bot";
  typing.innerHTML =
    '<div class="msg-av">AI</div><div class="msg-bub"><div class="typing"><div class="td"></div><div class="td"></div><div class="td"></div></div></div>';
  msgs.appendChild(typing);
  msgs.scrollTop = msgs.scrollHeight;

  setTimeout(() => {
    msgs.removeChild(typing);
    const b = document.createElement("div");
    b.className = "msg bot";
    b.innerHTML = `<div class="msg-av">AI</div><div class="msg-bub">${assistantReply(action)}</div>`;
    msgs.appendChild(b);
    msgs.scrollTop = msgs.scrollHeight;
  }, 1000);
}

function restartAll() {
  document.getElementById("chat-fab").classList.remove("show");
  chatOpen = false;
  document.getElementById("chat-panel").classList.remove("open");
  document.getElementById("text-input").value = "";
  onTextInput(document.getElementById("text-input"));
  document.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
  const audioInput = document.getElementById("audio-file");
  if (audioInput) audioInput.value = "";
  recordedAudioBlob = null;
  recordedAudioExt = "wav";
  recordedAudioLevel = 0;
  recordedChunks = [];
  pcmChunks = [];
  if (audioProcessorNode) {
    audioProcessorNode.disconnect();
    audioProcessorNode.onaudioprocess = null;
    audioProcessorNode = null;
  }
  if (audioSourceNode) {
    audioSourceNode.disconnect();
    audioSourceNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (recOn) toggleRec();
  recSecs = 0;
  clearInterval(recInterval);
  document.getElementById("timer").textContent = "00:00";
  document.getElementById("btn-audio-next").disabled = true;
  document.getElementById("status-txt").textContent = "Click mic to start recording";
  document.querySelectorAll(".analyzing-overlay").forEach((o) => o.classList.remove("show"));
  modalityScores.text = null;
  modalityScores.audio = null;
  modalityScores.video = null;
  modalityRaw.text = null;
  modalityRaw.audio = null;
  modalityRaw.video = null;
  apiErrors.text = null;
  apiErrors.audio = null;
  apiErrors.video = null;
  pendingPredictions.text = null;
  pendingPredictions.audio = null;
  pendingPredictions.video = null;
  stopCam();
  goToPage(0);
}


