export function clearHistoryState(state) {
  state.history.total = 0;
  state.history.items = [];
  state.history.selectedId = null;
  state.viewingHistoryId = null;
  state.savedHistoryId = null;
}

export function applyHistoryListPayload(state, payload) {
  state.history.items = Array.isArray(payload?.items) ? payload.items : [];
  state.history.total = Number(payload?.total || 0);
}

export function applyHistoryDetailPayload(state, payload) {
  state.history.selectedId = payload.id;
  state.viewingHistoryId = payload.id;
  state.savedHistoryId = payload.id;
  state.log = [];
  state.segments = [];
}
