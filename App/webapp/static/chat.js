document.addEventListener("DOMContentLoaded", () => {
  const openBtn = document.getElementById("chat-open");
  const panel = document.getElementById("chat-panel");
  const closeBtn = document.getElementById("chat-close");

  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messages = document.getElementById("chat-messages");

  const ctxEl = document.getElementById("chat-context");
  const ic = ctxEl?.dataset?.ic || "";
  const threshold = ctxEl?.dataset?.threshold || "";
  const gene = ctxEl?.dataset?.gene || "";

  function scrollToBottom() {
    requestAnimationFrame(() => {
      if (messages) {
        messages.scrollTop = messages.scrollHeight;
      }
    });
  }

  function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = `msg ${who}`;
    div.textContent = text;
    messages.appendChild(div);
    scrollToBottom();
  }

  // Start hidden
  if (panel) panel.classList.add("hidden");

  // Open chat
  if (openBtn && panel && input) {
    openBtn.addEventListener("click", (e) => {
      e.preventDefault();
      panel.classList.remove("hidden");
      input.focus();
      scrollToBottom();
    });
  }

  // Close chat
  if (closeBtn && panel) {
    closeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      panel.classList.add("hidden");
    });
  }

  // Submit message
  if (form && input && messages) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      let topPathways = [];
      try {
        const raw = ctxEl?.dataset?.topPathways || "[]";
        topPathways = JSON.parse(raw);
      } catch {
        topPathways = [];
      }

      const judgementNow =
        document.getElementById("llm-judgement-text")?.textContent?.trim() || "";

      addMessage(text, "user");
      input.value = "";

      // temporary loading bubble
      const loading = document.createElement("div");
      loading.className = "msg bot";
      loading.textContent = "...";
      messages.appendChild(loading);
      scrollToBottom();

      try {
        const resp = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            context: { ic, threshold, gene, judgement: judgementNow, topPathways }
          })
        });

        const data = await resp.json();

        // remove loading bubble
        messages.removeChild(loading);

        const reply = data.reply || "No reply.";

        // Clean error output (LLM not running etc.)
        if (reply.startsWith("Error:")) {
          addMessage(
            "Assistant is unavailable (LLM server not running).",
            "bot"
          );
        } else {
          addMessage(reply, "bot");
        }

      } catch (err) {
        messages.removeChild(loading);
        addMessage("Network error talking to the assistant.", "bot");
      }

      scrollToBottom();
    });
  }
});
