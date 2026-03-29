const API_BASE =
  window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;

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
const VIDEO_CAPTURE_MS = 5000;
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

function $(id) {
  return document.getElementById(id);
}

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
  $("page-" + curPage).classList.remove("active");
  curPage = n;
  $("page-" + n).classList.add("active");
  $("prog").style.width = progs[n] + "%";
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
  const cc = $("char-cnt");
  cc.textContent = l + " / 50";
  cc.className = "char-counter" + (l >= 50 ? " ok" : "");
  $("btn-text-next").disabled = l < 50;
}

function addChip(el, label) {
  el.classList.toggle("on");
  const ta = $("text-input");
  if (el.classList.contains("on")) {
    ta.value += (ta.value ? ", " : "") + label.toLowerCase();
    onTextInput(ta);
  }
}

function showOverlay(id, show) {
  const el = $(id);
  if (!el) return;
  if (show) el.classList.add("show");
  else el.classList.remove("show");
}

function parseApiScores(scores) {
  return {
    stress: scores.stress === undefined ? null : Number(scores.stress),
    anxiety: scores.anxiety === undefined ? null : Number(scores.anxiety),
    depression: scores.depression === undefined ? null : Number(scores.depression),
  };
}

async function analyzeText() {
  const text = $("text-input").value.trim();

  showOverlay("text-overlay", true);
  $("btn-text-next").disabled = true;

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
  const btn = $("btn-audio-next");
  if (input.files && input.files.length > 0) {
    btn.disabled = false;
    recordedAudioLevel = 1;
    $("status-txt").textContent = `Selected: ${input.files[0].name}`;
  }
}

let recOn = false;
let recInterval;
let recSecs = 0;
let micStream = null;
let recordedAudioBlob = null;
let recordedAudioExt = "wav";
let audioContext = null;
let audioSourceNode = null;
let audioProcessorNode = null;
let pcmChunks = [];
let recordedSampleRate = 44100;
let recordedAudioLevel = 0;
const wf = $("waveform");
for (let i = 0; i < 40; i++) {
  const b = document.createElement("div");
  b.className = "wb";
  b.style.animationDelay = i * 0.035 + "s";
  wf.appendChild(b);
}

function cleanupAudioRecordingResources() {
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
}

function resetWaveformBars(height = "4px") {
  wf.querySelectorAll(".wb").forEach((bar) => {
    bar.style.height = height;
  });
}

function setAudioRecordingUi(isRecording) {
  const btn = $("rec-btn");
  const zone = $("rec-zone");
  const tmr = $("timer");
  const st = $("rec-status");
  const stxt = $("status-txt");

  btn.classList.toggle("recording", isRecording);
  btn.textContent = isRecording ? "STOP" : "MIC";
  zone.classList.toggle("recording", isRecording);
  tmr.classList.toggle("live", isRecording);
  st.classList.toggle("live", isRecording);
  wf.classList.toggle("live", isRecording);
  stxt.textContent = isRecording ? "Recording in progress..." : `Complete - ${recSecs}s captured`;
}

async function toggleRec() {
  recOn = !recOn;
  const tmr = $("timer");
  const stxt = $("status-txt");

  if (recOn) {
    recordedAudioBlob = null;
    recordedAudioExt = "wav";
    recordedAudioLevel = 0;
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

    setAudioRecordingUi(true);
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
    cleanupAudioRecordingResources();
    setAudioRecordingUi(false);
    resetWaveformBars();
    clearInterval(recInterval);
    if (recSecs >= 5 && recordedAudioLevel >= 0.008) {
      $("btn-audio-next").disabled = false;
      stxt.textContent = `Complete - ${recSecs}s captured (ready to analyze)`;
    } else if (recSecs >= 5) {
      $("btn-audio-next").disabled = true;
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
  const input = $("audio-file");
  const hasUpload = input && input.files && input.files.length > 0;
  const hasRecording = recordedAudioBlob && recordedAudioBlob.size > 0;

  if (!hasRecording && !hasUpload) {
    alert("Please record audio first.");
    return;
  }

  showOverlay("audio-overlay", true);
  $("btn-audio-next").disabled = true;

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
let videoRecorder = null;

function getSupportedVideoMimeType() {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

function setCameraDecorations(enabled) {
  ["face-box", "scan-beam", "cc-tl", "cc-tr", "cc-bl", "cc-br"].forEach((id) =>
    $(id).classList.toggle("on", enabled)
  );
}

function setCameraUiActive(active) {
  $("cam-ph").style.display = active ? "none" : "";
  $("cam-start-btn").style.display = active ? "none" : "flex";
  $("cam-stop-btn").style.display = active ? "flex" : "none";
  $("btn-video-next").disabled = !active;
  setCameraDecorations(active);
  if (!active) {
    $("cam-start-btn").innerHTML = "Restart Camera";
    $("cam-ph").querySelector(".cam-placeholder-text").textContent = "Camera stopped";
  }
}

function cleanupVideoResources() {
  if (videoRecorder && videoRecorder.state !== "inactive") {
    videoRecorder.stop();
  }
  videoRecorder = null;
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
}

function startCam() {
  const onCameraReady = () => {
    setCameraUiActive(true);
    speakPrompt("Please place your face in front of the camera. A short video clip will be recorded for analysis.");
  };

  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((s) => {
      camStream = s;
      const v = $("cam-feed");
      v.srcObject = s;
      v.classList.add("on");
      onCameraReady();
    })
    .catch(() => {
      alert("Camera access failed. Please allow camera permission.");
    });
}

function stopCam() {
  cleanupVideoResources();
  $("cam-feed").classList.remove("on");
  setCameraUiActive(false);
}

async function recordCurrentVideoClip(durationMs) {
  if (!camStream) {
    throw new Error("Camera is not running.");
  }
  if (!window.MediaRecorder) {
    throw new Error("This browser does not support video recording.");
  }

  const mimeType = getSupportedVideoMimeType();
  if (!mimeType) {
    throw new Error("No supported browser video format was found.");
  }

  const chunks = [];

  return new Promise((resolve, reject) => {
    let settled = false;

    try {
      videoRecorder = new MediaRecorder(camStream, mimeType ? { mimeType } : undefined);
    } catch (err) {
      reject(err);
      return;
    }

    const finish = (blob) => {
      if (settled) return;
      settled = true;
      resolve(blob);
    };

    const fail = (err) => {
      if (settled) return;
      settled = true;
      reject(err);
    };

    videoRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    videoRecorder.onerror = () => fail(new Error("Video recording failed."));

    videoRecorder.onstop = () => {
      if (!chunks.length) {
        fail(new Error("No video data was recorded."));
        return;
      }
      const blob = new Blob(chunks, { type: mimeType || "video/webm" });
      finish(blob);
    };

    videoRecorder.start();
    setTimeout(() => {
      if (videoRecorder && videoRecorder.state !== "inactive") {
        videoRecorder.stop();
      }
    }, durationMs);
  });
}

async function analyzeVideo() {
  const video = $("cam-feed");
  if (!camStream || !video || video.readyState < 2) {
    speakPrompt("Please start the camera and wait for the live feed before recording the video clip.");
    alert("Please start the camera and wait for the live feed before analyzing video.");
    return;
  }

  showOverlay("video-overlay", true);
  $("btn-video-next").disabled = true;
  speakPrompt("Recording a short video clip. Please keep your face centered.");

  let clipBlob;
  try {
    clipBlob = await recordCurrentVideoClip(VIDEO_CAPTURE_MS);
  } catch (err) {
    showOverlay("video-overlay", false);
    $("btn-video-next").disabled = false;
    alert(err.message || "Video recording failed.");
    return;
  }

  pendingPredictions.video = (async () => {
    const form = new FormData();
    form.append("file", clipBlob, "camera-clip.webm");
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
      speakPrompt("No face was detected clearly in the recorded video. Please keep your face centered and try again.");
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
  fusedScores.stress = averageMetric("stress");
  fusedScores.anxiety = averageMetric("anxiety");
  fusedScores.depression = averageMetric("depression");
}

function averageMetric(metric) {
  const values = [modalityScores.text, modalityScores.audio, modalityScores.video]
    .filter(Boolean)
    .map((scores) => scores[metric])
    .filter((value) => value !== null && value !== undefined);

  if (values.length === 0) {
    return 0;
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  return Math.round(total / values.length);
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
  if (scores.depression !== null && scores.depression !== undefined) {
    items.push(explainSeverity(scores.depression, "Depression"));
  }
  items.push("This text branch contributes only to depression-related language assessment, not stress or anxiety scoring.");
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
      `The facial model saw ${raw.dominant_emotion} as the dominant visible emotion across the analyzed video frames.`
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
  const el = $("results-alert");
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
  const el = $(id);
  if (!el) return;
  el.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function animateMetric(arcId, valId, barId, target) {
  const arc = $(arcId);
  const val = $(valId);
  const bar = $(barId);

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
  $("chat-fab").classList.add("show");
  setTimeout(toggleChat, 800);
}

function toggleChat() {
  chatOpen = !chatOpen;
  const p = $("chat-panel");
  const n = $("chat-notif");
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
  const msgs = $("chat-msgs");
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

function resetPredictionState() {
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
}

function resetAudioUiAndState() {
  const audioInput = $("audio-file");
  if (audioInput) audioInput.value = "";
  recordedAudioBlob = null;
  recordedAudioExt = "wav";
  recordedAudioLevel = 0;
  pcmChunks = [];
  cleanupAudioRecordingResources();
  if (recOn) recOn = false;
  recSecs = 0;
  clearInterval(recInterval);
  $("timer").textContent = "00:00";
  $("btn-audio-next").disabled = true;
  $("status-txt").textContent = "Click mic to start recording";
  setAudioRecordingUi(false);
  resetWaveformBars();
}

function restartAll() {
  $("chat-fab").classList.remove("show");
  chatOpen = false;
  $("chat-panel").classList.remove("open");
  $("text-input").value = "";
  onTextInput($("text-input"));
  document.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
  resetAudioUiAndState();
  document.querySelectorAll(".analyzing-overlay").forEach((o) => o.classList.remove("show"));
  resetPredictionState();
  stopCam();
  goToPage(0);
}


