import { normalizeBannerType } from "./format.js";

export function createBannerController({ container: bannersContainerEl, document, storage = null }) {
  const dismissedBannerKeys = new Set();
  const storageKey = "whistx.dismissedAnnouncements.v1";
  try {
    const saved = JSON.parse(storage?.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) saved.filter(key => typeof key === "string").forEach(key => dismissedBannerKeys.add(key));
  } catch { /* Unavailable or corrupt storage must not prevent dismissal. */ }
  function renderBanners(rawBanners) {
    if (!bannersContainerEl) return;

    const banners = Array.isArray(rawBanners) ? rawBanners : [];
    bannersContainerEl.innerHTML = "";

    banners.forEach((banner, index) => {
      const record = banner && typeof banner === "object" ? banner : {};
      const id = String(record.id || `banner-${index + 1}`).trim() || `banner-${index + 1}`;
      const type = normalizeBannerType(record.type);
      const title = String(record.title || "").trim();
      const message = String(record.message || "").trim();
      const dismissible = record.dismissible !== false;

      if (!message) return;
      // Generated IDs follow list positions; reordering must not revive the
      // same content. Explicit IDs still distinguish separately issued notices.
      const identity = /^banner-\d+$/.test(id) ? "" : id;
      const dismissalKey = JSON.stringify([identity, type, title, message]);
      if (dismissible && dismissedBannerKeys.has(dismissalKey)) return;
      const node = document.createElement("article");
      node.className = `notice-banner notice-${type}`;
      node.setAttribute("role", "status");

      const header = document.createElement("div");
      header.className = "notice-banner-header";

      const heading = document.createElement("strong");
      heading.className = "notice-banner-title";
      heading.textContent = title || type.toUpperCase();
      header.appendChild(heading);

      if (dismissible) {
        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "notice-banner-close";
        closeBtn.setAttribute("aria-label", "バナーを閉じる");
        closeBtn.textContent = "×";
        closeBtn.addEventListener("click", () => {
          dismissedBannerKeys.add(dismissalKey);
          try { storage?.setItem(storageKey, JSON.stringify([...dismissedBannerKeys])); } catch { /* Keep the in-memory dismissal. */ }
          node.remove();
          bannersContainerEl.hidden = bannersContainerEl.childElementCount === 0;
        });
        header.appendChild(closeBtn);
      }

      const body = document.createElement("p");
      body.className = "notice-banner-body";
      body.textContent = message;

      node.appendChild(header);
      node.appendChild(body);
      bannersContainerEl.appendChild(node);
    });

    bannersContainerEl.hidden = bannersContainerEl.childElementCount === 0;
  }
  return { renderBanners };
}

export function createToastController({ container: toastContainer, document, schedule = setTimeout }) {
  // Toast notification system
  function showToast(message, type = "default", duration = 2500) {
    if (!toastContainer) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const iconEl = createToastIcon(type);
    if (iconEl) {
      toast.appendChild(iconEl);
    }
    const textEl = document.createElement("span");
    textEl.textContent = String(message || "");
    toast.appendChild(textEl);
    toastContainer.appendChild(toast);

    schedule(() => {
      toast.classList.add("hiding");
      schedule(() => toast.remove(), 250);
    }, duration);
  }

  function createToastIcon(type) {
    if (type !== "success" && type !== "error") {
      return null;
    }
    const svgNs = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("width", "16");
    svg.setAttribute("height", "16");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");

    if (type === "success") {
      const path = document.createElementNS(svgNs, "path");
      path.setAttribute("d", "M20 6L9 17l-5-5");
      svg.appendChild(path);
      return svg;
    }

    const circle = document.createElementNS(svgNs, "circle");
    circle.setAttribute("cx", "12");
    circle.setAttribute("cy", "12");
    circle.setAttribute("r", "10");
    const path1 = document.createElementNS(svgNs, "path");
    path1.setAttribute("d", "M15 9l-6 6");
    const path2 = document.createElementNS(svgNs, "path");
    path2.setAttribute("d", "M9 9l6 6");
    svg.append(circle, path1, path2);
    return svg;
  }
  return { showToast };
}
