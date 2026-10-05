export function createScreenshotViewerController(appDependencies) {
function clampScreenshotZoom(value) {
  return Math.min(appDependencies.SCREENSHOT_ZOOM_MAX, Math.max(appDependencies.SCREENSHOT_ZOOM_MIN, Number(value) || appDependencies.SCREENSHOT_ZOOM_MIN));
}

function updateScreenshotZoomUi() {
  if (!appDependencies.screenshotModalImageEl || !appDependencies.screenshotModalViewportEl || !appDependencies.screenshotModalStageEl) return;

  const zoom = clampScreenshotZoom(appDependencies.state.screenshotZoom);
  const viewportWidth = Math.max(1, appDependencies.screenshotModalViewportEl.clientWidth || 1);
  const viewportHeight = Math.max(1, appDependencies.screenshotModalViewportEl.clientHeight || 1);
  const baseWidth = Math.max(1, appDependencies.state.screenshotBaseWidth || viewportWidth);
  const baseHeight = Math.max(1, appDependencies.state.screenshotBaseHeight || viewportHeight);
  const renderedWidth = Math.max(1, Math.round(baseWidth * zoom));
  const renderedHeight = Math.max(1, Math.round(baseHeight * zoom));

  appDependencies.screenshotModalStageEl.style.width = Math.max(viewportWidth, renderedWidth) + "px";
  appDependencies.screenshotModalStageEl.style.height = Math.max(viewportHeight, renderedHeight) + "px";
  appDependencies.screenshotModalImageEl.style.width = renderedWidth + "px";
  appDependencies.screenshotModalImageEl.style.height = renderedHeight + "px";
  appDependencies.screenshotModalViewportEl.classList.toggle("is-zoomed", zoom > appDependencies.SCREENSHOT_ZOOM_MIN);
  appDependencies.screenshotModalViewportEl.classList.toggle("is-dragging", appDependencies.state.screenshotDragging);

  if (appDependencies.screenshotZoomOutBtnEl) appDependencies.screenshotZoomOutBtnEl.disabled = zoom <= appDependencies.SCREENSHOT_ZOOM_MIN;
  if (appDependencies.screenshotZoomInBtnEl) appDependencies.screenshotZoomInBtnEl.disabled = zoom >= appDependencies.SCREENSHOT_ZOOM_MAX;
  if (appDependencies.screenshotZoomResetBtnEl) appDependencies.screenshotZoomResetBtnEl.textContent = Math.round(zoom * 100) + "%";
}

function recalculateScreenshotBaseSize() {
  if (!appDependencies.screenshotModalImageEl || !appDependencies.screenshotModalViewportEl) return;
  if (!appDependencies.screenshotModalImageEl.naturalWidth || !appDependencies.screenshotModalImageEl.naturalHeight) return;

  const viewportWidth = Math.max(1, appDependencies.screenshotModalViewportEl.clientWidth || 1);
  const viewportHeight = Math.max(1, appDependencies.screenshotModalViewportEl.clientHeight || 1);
  const fitScale = Math.min(viewportWidth / appDependencies.screenshotModalImageEl.naturalWidth, viewportHeight / appDependencies.screenshotModalImageEl.naturalHeight);

  appDependencies.state.screenshotBaseWidth = Math.max(1, Math.round(appDependencies.screenshotModalImageEl.naturalWidth * fitScale));
  appDependencies.state.screenshotBaseHeight = Math.max(1, Math.round(appDependencies.screenshotModalImageEl.naturalHeight * fitScale));
}

function centerScreenshotViewport() {
  if (!appDependencies.screenshotModalViewportEl) return;
  const maxScrollLeft = Math.max(0, appDependencies.screenshotModalViewportEl.scrollWidth - appDependencies.screenshotModalViewportEl.clientWidth);
  const maxScrollTop = Math.max(0, appDependencies.screenshotModalViewportEl.scrollHeight - appDependencies.screenshotModalViewportEl.clientHeight);
  appDependencies.screenshotModalViewportEl.scrollLeft = maxScrollLeft / 2;
  appDependencies.screenshotModalViewportEl.scrollTop = maxScrollTop / 2;
}

function setScreenshotZoom(nextZoom, { recenter = false } = {}) {
  if (!appDependencies.screenshotModalViewportEl) return;

  const previousScrollableX = Math.max(0, appDependencies.screenshotModalViewportEl.scrollWidth - appDependencies.screenshotModalViewportEl.clientWidth);
  const previousScrollableY = Math.max(0, appDependencies.screenshotModalViewportEl.scrollHeight - appDependencies.screenshotModalViewportEl.clientHeight);
  const ratioX = previousScrollableX > 0 ? appDependencies.screenshotModalViewportEl.scrollLeft / previousScrollableX : 0.5;
  const ratioY = previousScrollableY > 0 ? appDependencies.screenshotModalViewportEl.scrollTop / previousScrollableY : 0.5;

  appDependencies.state.screenshotZoom = clampScreenshotZoom(nextZoom);
  updateScreenshotZoomUi();

  if (recenter) {
    centerScreenshotViewport();
    return;
  }

  const nextScrollableX = Math.max(0, appDependencies.screenshotModalViewportEl.scrollWidth - appDependencies.screenshotModalViewportEl.clientWidth);
  const nextScrollableY = Math.max(0, appDependencies.screenshotModalViewportEl.scrollHeight - appDependencies.screenshotModalViewportEl.clientHeight);
  appDependencies.screenshotModalViewportEl.scrollLeft = nextScrollableX * ratioX;
  appDependencies.screenshotModalViewportEl.scrollTop = nextScrollableY * ratioY;
}

function setScreenshotZoomAt(nextZoom, clientX, clientY) {
  if (!appDependencies.screenshotModalViewportEl) return;

  const currentZoom = clampScreenshotZoom(appDependencies.state.screenshotZoom);
  const resolvedZoom = clampScreenshotZoom(nextZoom);
  if (resolvedZoom === currentZoom) return;

  const rect = appDependencies.screenshotModalViewportEl.getBoundingClientRect();
  const offsetX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  const offsetY = Math.max(0, Math.min(rect.height, clientY - rect.top));
  const baseWidth = Math.max(1, appDependencies.state.screenshotBaseWidth || rect.width || 1);
  const baseHeight = Math.max(1, appDependencies.state.screenshotBaseHeight || rect.height || 1);
  const currentWidth = Math.max(1, Math.round(baseWidth * currentZoom));
  const currentHeight = Math.max(1, Math.round(baseHeight * currentZoom));
  const contentRatioX = (appDependencies.screenshotModalViewportEl.scrollLeft + offsetX) / currentWidth;
  const contentRatioY = (appDependencies.screenshotModalViewportEl.scrollTop + offsetY) / currentHeight;

  appDependencies.state.screenshotZoom = resolvedZoom;
  updateScreenshotZoomUi();

  if (resolvedZoom <= appDependencies.SCREENSHOT_ZOOM_MIN) {
    centerScreenshotViewport();
    return;
  }

  const nextWidth = Math.max(1, Math.round(baseWidth * resolvedZoom));
  const nextHeight = Math.max(1, Math.round(baseHeight * resolvedZoom));
  const nextScrollableX = Math.max(0, appDependencies.screenshotModalViewportEl.scrollWidth - appDependencies.screenshotModalViewportEl.clientWidth);
  const nextScrollableY = Math.max(0, appDependencies.screenshotModalViewportEl.scrollHeight - appDependencies.screenshotModalViewportEl.clientHeight);
  appDependencies.screenshotModalViewportEl.scrollLeft = Math.max(0, Math.min(nextScrollableX, contentRatioX * nextWidth - offsetX));
  appDependencies.screenshotModalViewportEl.scrollTop = Math.max(0, Math.min(nextScrollableY, contentRatioY * nextHeight - offsetY));
}

function zoomScreenshot(delta, anchorEvent = null) {
  if (anchorEvent && appDependencies.screenshotModalViewportEl) {
    setScreenshotZoomAt(appDependencies.state.screenshotZoom + delta, anchorEvent.clientX, anchorEvent.clientY);
    return;
  }
  setScreenshotZoom(appDependencies.state.screenshotZoom + delta);
}

function stopScreenshotDrag() {
  if (appDependencies.screenshotModalViewportEl && appDependencies.state.screenshotDragPointerId !== null) {
    try {
      if (appDependencies.screenshotModalViewportEl.hasPointerCapture?.(appDependencies.state.screenshotDragPointerId)) {
        appDependencies.screenshotModalViewportEl.releasePointerCapture(appDependencies.state.screenshotDragPointerId);
      }
    } catch {
      // ignore pointer capture release failures
    }
  }
  appDependencies.state.screenshotDragging = false;
  appDependencies.state.screenshotDragPointerId = null;
  updateScreenshotZoomUi();
}

function beginScreenshotDrag(event) {
  if (!appDependencies.screenshotModalViewportEl || appDependencies.state.screenshotZoom <= appDependencies.SCREENSHOT_ZOOM_MIN) return;
  appDependencies.state.screenshotDragging = true;
  appDependencies.state.screenshotDragPointerId = event.pointerId;
  appDependencies.state.screenshotDragStartX = event.clientX;
  appDependencies.state.screenshotDragStartY = event.clientY;
  appDependencies.state.screenshotDragScrollLeft = appDependencies.screenshotModalViewportEl.scrollLeft;
  appDependencies.state.screenshotDragScrollTop = appDependencies.screenshotModalViewportEl.scrollTop;
  appDependencies.screenshotModalViewportEl.setPointerCapture?.(event.pointerId);
  updateScreenshotZoomUi();
  event.preventDefault();
}

function handleScreenshotDrag(event) {
  if (!appDependencies.screenshotModalViewportEl || !appDependencies.state.screenshotDragging) return;
  if (appDependencies.state.screenshotDragPointerId !== null && event.pointerId !== appDependencies.state.screenshotDragPointerId) return;
  const deltaX = event.clientX - appDependencies.state.screenshotDragStartX;
  const deltaY = event.clientY - appDependencies.state.screenshotDragStartY;
  appDependencies.screenshotModalViewportEl.scrollLeft = appDependencies.state.screenshotDragScrollLeft - deltaX;
  appDependencies.screenshotModalViewportEl.scrollTop = appDependencies.state.screenshotDragScrollTop - deltaY;
  event.preventDefault();
}

function resetScreenshotZoom() {
  setScreenshotZoom(appDependencies.SCREENSHOT_ZOOM_MIN, { recenter: true });
}

function showScreenshotModal(src, alt = "スクリーンショット") {
  if (!appDependencies.runtimeUi.screenshotModalEl || !appDependencies.runtimeUi.screenshotModalImageEl) return;
  appDependencies.state.screenshotZoom = appDependencies.SCREENSHOT_ZOOM_MIN;
  appDependencies.state.screenshotBaseWidth = 0;
  appDependencies.state.screenshotBaseHeight = 0;
  stopScreenshotDrag();
  appDependencies.runtimeUi.screenshotModalImageEl.src = src;
  appDependencies.runtimeUi.screenshotModalImageEl.alt = alt;
  appDependencies.openManagedModal(appDependencies.runtimeUi.screenshotModalEl, { initialFocus: appDependencies.screenshotModalCloseEl });

  if (appDependencies.runtimeUi.screenshotModalImageEl.complete) {
    recalculateScreenshotBaseSize();
    resetScreenshotZoom();
  } else {
    updateScreenshotZoomUi();
  }
}

function hideScreenshotModal() {
  if (!appDependencies.runtimeUi.screenshotModalEl || !appDependencies.runtimeUi.screenshotModalImageEl) return;
  if (!appDependencies.closeManagedModal(appDependencies.runtimeUi.screenshotModalEl)) return;
  appDependencies.runtimeUi.screenshotModalImageEl.src = "";
  appDependencies.runtimeUi.screenshotModalImageEl.style.width = "";
  appDependencies.runtimeUi.screenshotModalImageEl.style.height = "";
  appDependencies.screenshotModalStageEl?.style.removeProperty("width");
  appDependencies.screenshotModalStageEl?.style.removeProperty("height");
  appDependencies.state.screenshotZoom = appDependencies.SCREENSHOT_ZOOM_MIN;
  appDependencies.state.screenshotBaseWidth = 0;
  appDependencies.state.screenshotBaseHeight = 0;
  stopScreenshotDrag();
  updateScreenshotZoomUi();
}

  return { clampScreenshotZoom, updateScreenshotZoomUi, recalculateScreenshotBaseSize, centerScreenshotViewport, setScreenshotZoom, setScreenshotZoomAt, zoomScreenshot, stopScreenshotDrag, beginScreenshotDrag, handleScreenshotDrag, resetScreenshotZoom, showScreenshotModal, hideScreenshotModal };
}
