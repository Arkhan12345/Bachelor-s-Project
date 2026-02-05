document.addEventListener("DOMContentLoaded", () => {
  const openBtn = document.getElementById("chat-open");   // your "Continue chat" button
  const panel = document.getElementById("chat-panel");
  const closeBtn = document.getElementById("chat-close");

  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messages = document.getElementById("chat-messages");

  function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = `msg ${who}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  // Start hidden every time
  if (panel) panel.classList.add("hidden");

  // Open on button click only
  if (openBtn && panel && input) {
    openBtn.addEventListener("click", (e) => {
      e.preventDefault();
      panel.classList.remove("hidden");
      input.focus();
    });
  }

  // Close button
  if (closeBtn && panel) {
    closeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      panel.classList.add("hidden");
    });
  }

  // Chat submit
  if (form && input && messages) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      addMessage(text, "user");
      input.value = "";

      addMessage("...", "bot");

      try {
        const resp = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });

        const data = await resp.json();
        messages.removeChild(messages.lastChild);
        addMessage(data.reply || "No reply.", "bot");
      } catch {
        messages.removeChild(messages.lastChild);
        addMessage("Network error talking to the assistant.", "bot");
      }
    });
  }
});
