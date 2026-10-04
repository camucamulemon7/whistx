import { transcriptParagraphs } from "../meeting/paragraphs.js";
import { formatTimestamp as formatMs } from "../ui/format.js";
import { transcriptJoiner } from "../meeting/paragraphs.js";

export function createTranscriptController(appDependencies) {
function renderEmptyTranscriptState() {
  appDependencies.logEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </div>
      <p class="empty-title">まだ文字起こしがありません</p>
      <p class="empty-description">録音を開始すると、リアルタイムで文字起こしが表示されます</p>
    </div>
  `;
}

function updateSegmentCount() {
  requestAnimationFrame(updateTranscriptLatestButton);
  const count = appDependencies.state.segments.length;
  appDependencies.segmentCountEl.textContent = `${count}件`;
  if (appDependencies.connCountEl) {
    appDependencies.connCountEl.textContent = String(count);
  }

  appDependencies.segmentCountEl.classList.remove("updated");
  void appDependencies.segmentCountEl.offsetWidth;
  appDependencies.segmentCountEl.classList.add("updated");
}

function extractTranscriptText() {
  const fromState = appDependencies.state.segments
    .map((segment) => renderTranscriptText(segment.text, segment.speaker))
    .join("\n")
    .trim();
  if (fromState) return fromState;

  const rows = Array.from(appDependencies.logEl.querySelectorAll(".log-row .text"));
  const fromDom = rows.map((node) => node.textContent || "").join("\n").trim();
  return fromDom;
}

function updateDownloadLinks() {
  appDependencies.meetingWorkspace?.syncSource();
  const links = [
    [appDependencies.dlTxt, "txt"],
    [appDependencies.dlJsonl, "jsonl"],
    [appDependencies.dlZip, "zip"],
  ];
  const locked = appDependencies.isRecordingInteractionLocked();
  const historyId = !locked ? appDependencies.state.viewingHistoryId : "";
  const runtimeId = !locked && appDependencies.state.runtimeSessionFinalized ? appDependencies.state.runtimeSessionId : "";
  const suffix = appDependencies.state.runtimeSessionToken ? `?token=${encodeURIComponent(appDependencies.state.runtimeSessionToken)}` : "";

  links.forEach(([link, extension]) => {
    const href = historyId
      ? `/api/history/${historyId}/download.${extension}`
      : runtimeId
        ? `/api/transcript/${runtimeId}.${extension}${suffix}`
        : "";
    if (href) {
      link.setAttribute("href", href);
      link.setAttribute("aria-disabled", "false");
      link.removeAttribute("tabindex");
      link.classList.remove("is-disabled");
      link.title = `${extension.toUpperCase()}を書き出す`;
      return;
    }
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    link.setAttribute("tabindex", "-1");
    link.classList.add("is-disabled");
    link.classList.remove("is-downloaded");
    link.title = locked ? "録音の完了後に書き出せます" : "書き出せる文字起こしがありません";
  });

  const exportMenu = appDependencies.dlTxt?.closest(".export-menu");
  exportMenu?.classList.toggle("is-disabled", !historyId && !runtimeId);
  exportMenu?.querySelector("summary")?.setAttribute("aria-disabled", String(!historyId && !runtimeId));
}

function renderTranscriptText(rawText, speaker) {
  const clean = String(rawText || "").trim();
  if (!clean) return "";
  const label = String(speaker || "").trim();
  if (!label) return clean;
  return `[${label}] ${clean}`;
}

function isLogNearBottom() {
  if (!appDependencies.logEl) return true;
  const distance = appDependencies.logEl.scrollHeight - appDependencies.logEl.clientHeight - appDependencies.logEl.scrollTop;
  return distance <= 48;
}

function scrollLogToBottom() {
  if (!appDependencies.logEl) return;
  requestAnimationFrame(() => {
    appDependencies.logEl.scrollTop = appDependencies.logEl.scrollHeight;
    updateTranscriptLatestButton();
  });
}

function updateTranscriptLatestButton() {
  document.querySelector("#transcriptLatestBtn").hidden = !appDependencies.state.segments.length || isLogNearBottom();
}

function renderTranscriptParagraphs() {
  const nodes = new Map([...appDependencies.logEl.querySelectorAll(".log-row")].map(node => [node.dataset.segmentId, node]));
  const oldGroups = new Map([...appDependencies.logEl.querySelectorAll(".transcript-paragraph")].map(node => [node.dataset.firstSegment, node]));
  for (const segments of transcriptParagraphs(appDependencies.state.segments)) {
    const first = segments[0];
    const key = String(first.segmentId || first.seq);
    const paragraph = oldGroups.get(key) || document.createElement("section");
    oldGroups.delete(key);
    paragraph.className = "transcript-paragraph";
    paragraph.dataset.firstSegment = key;
    let heading = paragraph.querySelector(".transcript-paragraph-heading");
    if (!heading) { heading = document.createElement("div"); heading.className = "transcript-paragraph-heading"; paragraph.prepend(heading); }
    heading.textContent = `${formatMs(first.tsStart)}${first.speaker ? ` · ${first.speaker}` : first.track === "display" ? " · 共有音声" : ""}`;
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index];
      const row = nodes.get(String(segment.segmentId || segment.seq));
      if (!row) continue;
      const text = row.querySelector(".text");
      text.textContent = (index ? transcriptJoiner(segments[index - 1].text, segment.text) : "") + segment.text;
      paragraph.append(row);
    }
    appDependencies.logEl.append(paragraph);
  }
  for (const paragraph of oldGroups.values()) paragraph.remove();
}

function addLogLine(text, tsStart, tsEnd, seq, speaker, screenshotPath = "", rawAudioPath = "", audioPath = "", segmentId = "", metadata = {}) {
  if (segmentId && appDependencies.state.segments.some((item) => item.segmentId === segmentId)) return;
  const shouldAutoScroll = appDependencies.state.logAutoScrollEnabled || isLogNearBottom();

  // Hide empty state on first log entry
  const emptyState = appDependencies.logEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }

  const row = document.createElement("div");
  row.className = "log-row new";
  row.dataset.segmentId = segmentId || String(seq);
  row.dataset.quality = metadata.quality || "";
  if (metadata.quality) {
    const quality = document.createElement("span");
    quality.className = "transcript-quality";
    quality.dataset.quality = metadata.quality;
    quality.textContent = metadata.quality === "high_accuracy"
      ? (metadata.retainedRealtimeSegmentIds?.length ? "高精度・一部速報を保持" : "高精度") : "リアルタイム";
    row.append(quality);
  }
  row.tabIndex = -1;

  const range = document.createElement("span");
  range.className = "time";
  range.textContent = `${formatMs(tsStart)} - ${formatMs(tsEnd)}`;

  const content = document.createElement("span");
  content.className = "text";
  content.dataset.rawText = text;
  if (speaker) {
    content.dataset.speaker = speaker;
  }
  content.textContent = renderTranscriptText(text, speaker);

  if (Number.isFinite(Number(seq))) {
    row.dataset.seq = String(Number(seq));
  }

  const body = document.createElement("div");
  body.className = "log-body";
  body.append(content);

  let mediaGroup = null;
  if (screenshotPath || rawAudioPath || audioPath) {
    mediaGroup = document.createElement("div");
    mediaGroup.className = "log-media-group";
    mediaGroup.dataset.hasScreenshot = screenshotPath ? "true" : "false";
    mediaGroup.dataset.hasAudio = (rawAudioPath || audioPath) ? "true" : "false";
  }

  if (screenshotPath && mediaGroup) {
    const link = document.createElement("a");
    link.className = "log-screenshot-link";
    link.href = screenshotPath;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = "スクリーンショットを開く";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      appDependencies.showScreenshotModal(screenshotPath, `${formatMs(tsStart)} の画面キャプチャ`);
    });

    link.textContent = "共有画面を表示 ↗";
    mediaGroup.append(link);
  }

  if ((rawAudioPath || audioPath) && mediaGroup) {
    const audioActions = document.createElement("div");
    audioActions.className = "log-audio-actions";
    const inlineAudio = document.createElement("audio");
    inlineAudio.className = "log-inline-audio";
    inlineAudio.controls = true;
    inlineAudio.preload = "none";
    inlineAudio.hidden = true;
    inlineAudio.addEventListener("play", () => {
      document.querySelectorAll(".log-inline-audio").forEach((element) => {
        if (element !== inlineAudio && typeof element.pause === "function") {
          element.pause();
        }
      });
    });

    const createAudioButton = (label, url, extraClass = "") => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `log-audio-link ${extraClass}`.trim();
      button.title = `${label}を再生`;
      button.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="6 3 20 12 6 21 6 3"></polygon>
        </svg>
        <span>${label}</span>
      `;
      button.addEventListener("click", async () => {
        const isSameSource = inlineAudio.dataset.src === url;
        if (isSameSource && !inlineAudio.hidden) {
          if (inlineAudio.paused) {
            try {
              await inlineAudio.play();
            } catch {
              // ignore
            }
          } else {
            inlineAudio.pause();
          }
          return;
        }
        inlineAudio.pause();
        inlineAudio.src = url;
        inlineAudio.dataset.src = url;
        inlineAudio.hidden = false;
        try {
          await inlineAudio.play();
        } catch {
          // controls stay visible for manual playback
        }
      });
      return button;
    };

    if (rawAudioPath) {
      audioActions.append(createAudioButton("元音声", rawAudioPath));
    }

    if (audioPath) {
      audioActions.append(createAudioButton("加工後", audioPath, "is-processed"));
    }

    mediaGroup.append(audioActions);
    mediaGroup.append(inlineAudio);
  }

  if (mediaGroup) {
    body.append(mediaGroup);
  }

  row.append(range, body);
  appDependencies.logEl.appendChild(row);
  if (shouldAutoScroll) {
    scrollLogToBottom();
  }

  appDependencies.state.log.push(text);
  appDependencies.state.segments.push({
    ...metadata,
    segmentId: segmentId || String(seq),
    text,
    tsStart,
    tsEnd,
    seq,
    speaker,
    screenshotPath,
    rawAudioPath,
    audioPath,
  });
  renderTranscriptParagraphs();
  appDependencies.markWorkspaceDirty();
  updateSegmentCount();
  appDependencies.markProofreadStale();
  appDependencies.updateSaveControls();

  // Remove 'new' class after animation
  setTimeout(() => row.classList.remove("new"), 300);
}

function applySpeakerPatch(segments) {
  if (!Array.isArray(segments) || !segments.length) return;

  for (const seg of segments) {
    const seq = Number(seg?.seq);
    const speaker = String(seg?.speaker || "").trim();
    if (!Number.isFinite(seq) || !speaker) continue;

    const row = appDependencies.logEl.querySelector(`.log-row[data-seq="${seq}"]`);
    if (!row) continue;

    const textNode = row.querySelector(".text");
    if (!textNode) continue;

    const raw = String(textNode.dataset.rawText || textNode.textContent || "").trim();
    textNode.dataset.rawText = raw;
    textNode.dataset.speaker = speaker;
    textNode.textContent = renderTranscriptText(raw, speaker);

    const target = appDependencies.state.segments.find((item) => Number(item.seq) === seq);
    if (target) {
      target.speaker = speaker;
    }
  }
  renderTranscriptParagraphs();
}

async function copyAll() {
  const text = extractTranscriptText();
  if (!text) {
    appDependencies.showToast("コピーする内容がありません", "error");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    appDependencies.copyBtn.classList.add("is-success");
    appDependencies.showToast("コピーしました", "success");
    appDependencies.setStatus("copied");
    setTimeout(() => appDependencies.copyBtn.classList.remove("is-success"), 1200);
  } catch {
    appDependencies.showToast("コピーに失敗しました", "error");
    appDependencies.setStatus("copy_failed");
  }
}

  return { renderEmptyTranscriptState, updateSegmentCount, extractTranscriptText, updateDownloadLinks, renderTranscriptText, isLogNearBottom, scrollLogToBottom, updateTranscriptLatestButton, renderTranscriptParagraphs, addLogLine, applySpeakerPatch, copyAll };
}
