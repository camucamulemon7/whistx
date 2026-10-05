export function createModalController({ document, window, getBlockingModals }) {
  let modalStack = [];
  const modalOpeners = new WeakMap();
  function hasOpenBlockingModal() {
    return getBlockingModals().some((element) => element && !element.hidden);
  }

  function syncBodyScrollLock() {
    const locked = hasOpenBlockingModal();
    document.body.style.overflow = locked ? "hidden" : "";
    document.body.classList.toggle("is-modal-open", locked);
  }

  function lockBodyScroll() {
    syncBodyScrollLock();
  }

  function unlockBodyScroll() {
    syncBodyScrollLock();
  }

  function modalFocusableElements(modal) {
    if (!modal) return [];
    return Array.from(
      modal.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), iframe, [tabindex]:not([tabindex="-1"])'
      )
    ).filter((element) => {
      const style = window.getComputedStyle(element);
      return !element.hidden && style.display !== "none" && style.visibility !== "hidden";
    });
  }

  function topmostModal() {
    for (let index = modalStack.length - 1; index >= 0; index -= 1) {
      const modal = modalStack[index];
      if (modal && !modal.hidden) return modal;
    }
    return null;
  }

  function syncManagedModalLayers() {
    modalStack = modalStack.filter((modal) => modal && !modal.hidden);
    const top = modalStack[modalStack.length - 1] || null;
    modalStack.forEach((modal, index) => {
      modal.style.zIndex = String(31 + index);
      const inactive = modal !== top;
      modal.inert = inactive;
      if (inactive) {
        modal.setAttribute("aria-hidden", "true");
      } else {
        modal.removeAttribute("aria-hidden");
      }
    });
  }

  function openManagedModal(modal, options = {}) {
    if (!modal) return;
    const opener = options.opener || document.activeElement;
    if (opener instanceof window.HTMLElement) {
      modalOpeners.set(modal, opener);
    }
    modalStack = modalStack.filter((item) => item !== modal);
    modalStack.push(modal);
    modal.hidden = false;
    modal.classList.add("is-open");
    syncManagedModalLayers();
    lockBodyScroll();
    window.requestAnimationFrame(() => {
      const target =
        options.initialFocus ||
        modalFocusableElements(modal)[0] ||
        modal.querySelector('[role="dialog"]');
      target?.focus?.();
    });
  }

  function closeManagedModal(modal) {
    if (!modal || modal.hidden) return false;
    const wasTopmost = topmostModal() === modal;
    modal.classList.remove("is-open");
    modal.hidden = true;
    modal.inert = false;
    modal.removeAttribute("aria-hidden");
    modal.style.removeProperty("z-index");
    modalStack = modalStack.filter((item) => item !== modal);
    syncManagedModalLayers();
    unlockBodyScroll();

    if (wasTopmost) {
      const nextModal = topmostModal();
      if (nextModal) {
        const nextTarget = modalFocusableElements(nextModal)[0] || nextModal.querySelector('[role="dialog"]');
        nextTarget?.focus?.();
      } else {
        const opener = modalOpeners.get(modal);
        if (opener?.isConnected && !opener.disabled) {
          opener.focus();
        }
      }
    }
    modalOpeners.delete(modal);
    return true;
  }

  function trapModalFocus(event, modal) {
    if (event.key !== "Tab" || !modal) return;
    const focusable = modalFocusableElements(modal);
    if (focusable.length === 0) {
      event.preventDefault();
      modal.querySelector('[role="dialog"]')?.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !modal.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !modal.contains(document.activeElement))) {
      event.preventDefault();
      first.focus();
    }
  }

  return { openManagedModal, closeManagedModal, topmostModal, trapModalFocus };
}
