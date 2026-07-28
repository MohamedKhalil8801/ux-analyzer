(() => {
  const accountToggle = document.querySelector("[data-account-toggle]");
  const accountPanel = document.querySelector("[data-account-panel]");

  if (accountToggle instanceof HTMLButtonElement && accountPanel instanceof HTMLElement) {
    accountToggle.addEventListener("click", () => {
      accountPanel.hidden = !accountPanel.hidden;
    });
  }

  document.querySelectorAll("[data-dialog-target]").forEach((trigger) => {
    if (!(trigger instanceof HTMLElement)) {
      return;
    }
    const dialogId = trigger.dataset.dialogTarget;
    const dialog = dialogId ? document.getElementById(dialogId) : null;
    if (!(dialog instanceof HTMLDialogElement)) {
      return;
    }
    trigger.addEventListener("click", () => dialog.showModal());
  });

  document.querySelectorAll("[data-dialog-close]").forEach((closeButton) => {
    closeButton.addEventListener("click", () => {
      const dialog = closeButton.closest("dialog");
      if (dialog instanceof HTMLDialogElement) {
        dialog.close();
      }
    });
  });
})();
