import { fetchJson } from "../api/client.js";
import { readSseJsonStream } from "../api/sse.js";
import { escapeHtml, formatTimestamp } from "../ui/format.js";

function element(tag, className, text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

export function safeMediaUrl(value) {
  if (typeof value !== "string" || !value.startsWith("/api/")) return "";
  try {
    const url = new URL(value, location.origin);
    return url.origin === location.origin && /^\/api\/(history|transcripts)\/[^/]+\/(screenshots|audio)\/[^/]+$/.test(url.pathname) ? url.pathname : "";
  } catch { return ""; }
}

export function createMeetingWorkspace({ getSource, getAccess = () => "ready", onNavigate, onRecap, onImage, onBusy }) {
  const panels = document.querySelector("#workspacePanels");
  const recapRoot = document.querySelector("#summaryText");
  const turnsRoot = document.querySelector("#assistantTurns");
  const status = document.querySelector("#assistantStatus");
  const question = document.querySelector("#assistantQuestion");
  const askButton = document.querySelector("#assistantAsk");
  const summaryButton = document.querySelector("#summaryBtn");
  const summaryLabel = document.querySelector("#summaryBtnLabel");
  const summaryMeta = document.querySelector("#summaryMeta");
  let sourceKey = "";
  let generation = 0;
  let payload = null;
  let loadController = null;
  let recapController = null;
  let answerController = null;
  let view = "transcript";
  let completedTurns = [];

  function setView(next) {
    view = next;
    panels.dataset.meetingView = next;
    document.querySelectorAll("[data-meeting-tab]").forEach((button) => {
      const selected = button.dataset.meetingTab === next;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  }
  document.querySelectorAll("[data-meeting-tab]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.meetingTab));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const buttons = [...document.querySelectorAll("[data-meeting-tab]")].filter((item) => item.getClientRects().length);
      let index = buttons.indexOf(button);
      index = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[index].click();
      buttons[index].focus();
    });
  });
  panels.classList.add("meeting-layout");
  setView(view);

  function busy() {
    const any = Boolean(recapController || answerController);
    onBusy?.(any);
    askButton.textContent = answerController ? "キャンセル" : "送信";
    askButton.setAttribute("aria-busy", String(Boolean(answerController)));
    summaryButton.setAttribute("aria-busy", String(Boolean(recapController)));
    summaryLabel.textContent = recapController ? "キャンセル" : "要約する";
    updateAccess();
  }

  function updateAccess() {
    const access = getAccess();
    const message = access === "login" ? "会議アシスタントにはログインが必要です。ゲストでは文字起こしのみ利用できます。"
      : access === "unavailable" ? "会議アシスタントを利用できません。サーバーの機能設定を確認してください。" : "";
    const notice = document.querySelector("#assistantAccessNotice");
    notice.hidden = !message;
    notice.textContent = message;
    question.disabled = Boolean(message);
    askButton.disabled = Boolean(message) && !answerController;
    document.querySelectorAll("[data-meeting-question]").forEach(button => { button.disabled = Boolean(message); });
  }

  function sourceButton(source, label = "") {
    const button = element("button", "meeting-citation", label || formatTimestamp(source.startMs));
    button.type = "button";
    button.title = source.text;
    button.addEventListener("click", () => {
      setView("transcript");
      onNavigate(source);
    });
    return button;
  }

  function renderRecap(recap) {
    if (!recap) return;
    recapRoot.replaceChildren();
    summaryMeta.textContent = recap.stale ? "更新あり" : recap.provisional ? "ここまでの要約" : "会議の要約";
    const note = element("p", "meeting-note", `${formatTimestamp(recap.throughMs)} までの発話をもとに作成${recap.stale ? " · 新しい発話があります。再生成で更新できます。" : ""}`);
    recapRoot.append(note);
    const sources = new Map((recap.sources || []).map((row) => [row.id, row]));
    const images = new Map((payload?.images || []).map((image) => [image.id, image]));
    const index = element("nav", "chapter-index");
    index.setAttribute("aria-label", "会議の章");
    recapRoot.append(index);
    for (const chapter of recap.chapters || []) {
      const card = element("article", "chapter-card");
      card.id = `meeting-${chapter.id}`;
      const link = element("button", "chapter-index-link", chapter.title);
      link.type = "button";
      link.addEventListener("click", () => card.scrollIntoView({ behavior: "smooth", block: "start" }));
      index.append(link);
      const heading = element("div", "chapter-heading");
      heading.append(element("h3", "", chapter.title));
      const first = sources.get(chapter.sourceIds?.[0]);
      if (first) heading.append(sourceButton(first, `${formatTimestamp(chapter.startMs)}–${formatTimestamp(chapter.endMs)}`));
      card.append(heading);
      for (const [field, title] of [["summary", "要点"], ["decisions", "決定事項"], ["actions", "宿題"], ["open_questions", "未決事項"]]) {
        if (!chapter[field]?.length) continue;
        card.append(element("h4", "chapter-section-title", title));
        const list = element("ul", "chapter-facts");
        for (const fact of chapter[field]) {
          const item = element("li", "");
          item.append(document.createTextNode(fact.text));
          if (field === "actions") item.append(element("span", "action-owner", `担当: ${fact.owner || "未定"} · 期限: ${fact.due || "未定"}`));
          for (const id of fact.sourceIds || []) {
            if (sources.has(id)) item.append(sourceButton(sources.get(id)));
          }
          list.append(item);
        }
        card.append(list);
      }
      const frames = element("div", "chapter-images");
      for (const id of chapter.imageIds || []) {
        const image = images.get(id);
        if (image) frames.append(imageCard(image));
      }
      if (frames.childElementCount) card.append(frames);
      recapRoot.append(card);
    }
    document.querySelector("#exportRecap").disabled = !recap.chapters?.length;
  }

  function imageCard(image) {
    const figure = element("figure", "meeting-image");
    const button = element("button", "meeting-image-button");
    button.type = "button";
    const img = document.createElement("img");
    img.src = safeMediaUrl(image.url);
    img.alt = `${formatTimestamp(image.timeMs)} の共有画面`;
    img.loading = "lazy";
    button.append(img);
    button.addEventListener("click", () => onImage(img.src, img.alt));
    figure.append(button, element("figcaption", "", formatTimestamp(image.timeMs)));
    return figure;
  }

  function renderMaterials() {
    const root = document.querySelector("#meetingImages");
    document.querySelector("#meetingImageCount").textContent = `${payload?.images?.length || 0} 枚`;
    root.replaceChildren();
    for (const image of payload?.images || []) root.append(imageCard(image));
    if (!root.childElementCount) root.append(element("p", "meeting-note", "画面共有中に取得した画像がここに並びます。"));
  }

  function renderTurn(turn, draft = false) {
    const item = element("article", "assistant-turn");
    item.append(element("p", "assistant-question", turn.question));
    const answer = element("div", "assistant-answer", turn.answer || "");
    item.append(answer);
    if (!draft) {
      const refs = element("div", "assistant-citations");
      for (const citation of turn.citations || []) refs.append(sourceButton(citation, `[${citation.label}] ${formatTimestamp(citation.startMs)}`));
      item.append(refs, element("p", "meeting-note", `${formatTimestamp(turn.throughMs)} までの内容`));
    }
    turnsRoot.append(item);
    return { item, answer };
  }

  function renderTurns() {
    turnsRoot.replaceChildren();
    for (const turn of payload?.turns || []) renderTurn(turn);
    if (!turnsRoot.childElementCount) turnsRoot.append(element("p", "assistant-empty", "この会議について質問できます。回答の時刻を押すと、根拠の発話を確認できます。"));
  }

  function snapshotSource() {
    const source = getSource();
    if (!source) throw new Error("meeting_not_started");
    return source;
  }

  async function refresh() {
    const source = getSource();
    if (!source) return;
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    const version = generation;
    try {
      const result = await fetchJson("/api/meeting/insights", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(source), signal: controller.signal });
      if (version !== generation || controller.signal.aborted) return;
      // A response started before an answer completed must not erase that answer.
      const turnKey = (turn) => turn.id || JSON.stringify([turn.createdAt, turn.question, turn.answer, turn.throughMs]);
      const turns = new Map((result.turns || []).map((turn) => [turnKey(turn), turn]));
      for (const turn of completedTurns) turns.set(turnKey(turn), turn);
      const images = new Map((result.images || []).map((frame) => [frame.id, frame]));
      for (const frame of payload?.images || []) images.set(frame.id, frame);
      payload = { ...result, turns: [...turns.values()], images: [...images.values()] };
      if (!recapController) renderRecap(result.recap);
      if (!answerController) renderTurns();
      renderMaterials();
    } catch (error) {
      if (error.code !== "aborted" && version === generation) status.textContent = error.status === 401 ? "会議アシスタントにはログインが必要です" : "会議データを取得できませんでした";
    } finally {
      if (loadController === controller) loadController = null;
    }
  }

  function syncSource() {
    updateAccess();
    const key = JSON.stringify(getSource());
    if (key === sourceKey) return;
    sourceKey = key;
    generation += 1;
    loadController?.abort();
    recapController?.abort();
    answerController?.abort();
    loadController = recapController = answerController = null;
    payload = null;
    completedTurns = [];
    document.querySelector("#assistantLiveContext").textContent = "録音中も、ここまでの発話について質問できます";
    question.value = "";
    status.textContent = getSource() ? "この会議の発話を参照します" : "録音を開始するか、保存済みの会議を開いてください";
    renderTurns();
    renderMaterials();
    busy();
    document.querySelector("#exportRecap").disabled = true;
    refresh();
  }

  async function generateRecap() {
    if (recapController) { recapController.abort(); return; }
    let source;
    try { source = snapshotSource(); } catch { summaryMeta.textContent = "録音を開始してください"; return; }
    const version = generation;
    const controller = new AbortController();
    recapController = controller;
    busy();
    summaryMeta.textContent = "章ごとに整理しています…";
    setView("summary");
    try {
      const result = await fetchJson("/api/meeting/recap", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...source, prompt: document.querySelector("#summaryPrompt")?.value || "" }), signal: controller.signal, timeoutMs: 300_000 });
      if (version !== generation || controller.signal.aborted) return;
      payload = { ...payload, ...result };
      onRecap(result.summary, "会議の要約");
      renderRecap(result.recap);
      renderMaterials();
    } catch (error) {
      if (version === generation) summaryMeta.textContent = error.code === "aborted" ? "要約キャンセル" : error.message === "empty_transcript" ? "確定した発話を待っています" : "要約に失敗しました。再試行できます";
    } finally {
      if (recapController === controller) { recapController = null; busy(); }
    }
  }

  async function ask() {
    if (answerController) { answerController.abort(); return; }
    if (getAccess() !== "ready") { updateAccess(); return; }
    const text = question.value.trim();
    if (!text) { question.focus(); return; }
    let source;
    try { source = snapshotSource(); } catch { status.textContent = "録音を開始するか、会議を開いてください"; return; }
    const controller = new AbortController();
    answerController = controller;
    const version = generation;
    const timer = setTimeout(() => controller.abort("timeout"), 120_000);
    busy();
    turnsRoot.querySelector(".assistant-empty")?.remove();
    const draft = renderTurn({ question: text }, true);
    let done = false;
    status.textContent = "会議の内容を確認しています…";
    try {
      const response = await fetch("/api/meeting/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...source, question: text }), signal: controller.signal });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `meeting_http_${response.status}`);
      }
      await readSseJsonStream(response, (event) => {
        if (version !== generation || controller.signal.aborted) return;
        const follow = turnsRoot.scrollHeight - turnsRoot.scrollTop - turnsRoot.clientHeight < 120;
        if (event.type === "status") status.textContent = `${event.message} · ${formatTimestamp(event.throughMs)} まで`;
        if (event.type === "delta") draft.answer.append(document.createTextNode(event.text));
        if (event.type === "error") throw new Error(event.error);
        if (event.type === "done") {
          done = true;
          completedTurns.push(event);
          payload = { ...payload, turns: [...(payload?.turns || []), event] };
          draft.item.remove();
          renderTurn(event);
          question.value = "";
          status.textContent = "回答の出典から発話を確認できます";
        }
        if (follow) turnsRoot.scrollTop = turnsRoot.scrollHeight;
      });
      if (!done && !controller.signal.aborted) throw new Error("incomplete_answer");
    } catch (error) {
      if (version === generation) {
        const messages = {
          empty_transcript: "最初の発話を認識しています。文字起こしが表示されたら、録音中のまま質問できます。",
          summary_not_configured: "会議アシスタントのモデルが未設定です。管理者に要約モデルの設定を依頼してください。",
          meeting_model_unavailable: "回答用モデルに接続できません。モデルの接続設定を確認してください。",
          meeting_model_busy: "回答用モデルが処理中です。少し待ってから再試行してください。",
          rate_limit_exceeded: "質問が集中しています。少し待ってから再試行してください。",
          meeting_http_401: "会議アシスタントにはログインが必要です。",
        };
        draft.answer.textContent = controller.signal.aborted ? "回答をキャンセルしました。" : messages[error.message] || "回答を取得できませんでした。再試行してください。";
        status.textContent = draft.answer.textContent;
      }
    } finally {
      clearTimeout(timer);
      if (answerController === controller) { answerController = null; busy(); }
    }
  }
  document.querySelector("#assistantForm").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
  question.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); ask(); } });
  document.querySelectorAll("[data-meeting-question]").forEach((button) => button.addEventListener("click", () => {
    if (answerController) return;
    question.value = button.dataset.meetingQuestion;
    ask();
  }));

  async function exportRecap() {
    if (!payload?.recap) return;
    const button = document.querySelector("#exportRecap");
    button.disabled = true;
    try {
      const clone = recapRoot.cloneNode(true);
      clone.querySelector(".chapter-index")?.remove();
      for (const image of clone.querySelectorAll("img")) {
        const response = await fetch(safeMediaUrl(image.getAttribute("src")));
        if (!response.ok) throw new Error("image_download_failed");
        const blob = await response.blob();
        image.src = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      }
      for (const node of clone.querySelectorAll("button")) {
        const span = document.createElement("span");
        span.append(...node.childNodes);
        node.replaceWith(span);
      }
      const sourceText = (payload.recap.sources || []).map((row) => `<p><small>${escapeHtml(formatTimestamp(row.startMs))}</small> ${escapeHtml(row.text)}</p>`).join("");
      const html = `<!doctype html><html lang="ja"><meta charset="utf-8"><title>会議の要約</title><style>body{font:16px/1.8 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#202124}img{max-width:100%;height:auto}article{border-top:1px solid #ddd;padding:20px 0}figure{margin:16px 0}h4{margin-bottom:4px}.meeting-citation,small,figcaption{color:#555;font-size:12px;margin-left:8px}.action-owner{display:block;color:#555}</style><body><h1>会議の要約</h1>${clone.innerHTML}<h2>出典の文字起こし</h2>${sourceText}</body></html>`;
      const url = URL.createObjectURL(new Blob([html], { type: "text/html;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = "meeting-recap.html";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      summaryMeta.textContent = "画像付き保存に失敗しました。再試行できます";
    } finally { button.disabled = false; }
  }
  document.querySelector("#exportRecap").addEventListener("click", exportRecap);

  function liveEvent(event) {
    if (["final", "transcript_revision", "transcript_snapshot"].includes(event.type)) {
      const through = event.tsEnd || event.record?.tsEnd || event.records?.at(-1)?.tsEnd;
      if (through) document.querySelector("#assistantLiveContext").textContent = `${formatTimestamp(through)} までの発話を参照できます · 録音中も質問可能`;
    }
    if (event.type === "partial") {
      const root = document.querySelector("#liveTranscript");
      let row = [...root.children].find((item) => item.dataset.track === event.track);
      if (!event.text) { row?.remove(); root.hidden = !root.childElementCount; return; }
      if (!row) { row = element("div", "live-utterance"); row.dataset.track = event.track; root.append(row); }
      row.replaceChildren(element("span", "live-speaker", event.speaker || "認識中"));
      const text = element("span", "live-hypothesis");
      const stable = event.text.startsWith(event.stableText || "") ? event.stableText || "" : "";
      text.append(element("span", "live-stable", stable), document.createTextNode(event.text.slice(stable.length)));
      row.append(text);
      root.hidden = false;
      document.querySelector("#liveConnection").textContent = event.backlogMs > 3000 ? `認識待ち ${Math.round(event.backlogMs / 1000)}秒` : "ライブ文字起こし";
    } else if (event.type === "screen") {
      if (!payload) payload = { images: [] };
      const id = event.screenshotPath.split("/").pop();
      payload.images ||= [];
      if (!payload.images.some((image) => image.id === id)) payload.images.push({ id, timeMs: event.timeMs, url: event.screenshotPath });
      renderMaterials();
    } else if (event.type === "connection" || event.type === "backpressure") {
      document.querySelector("#liveConnection").textContent = event.message;
    }
  }

  return { syncSource, refresh, generateRecap, liveEvent, setView, get busy() { return Boolean(recapController || answerController); } };
}
