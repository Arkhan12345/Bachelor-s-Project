const toggleBtn = document.getElementById("chat-toggle");
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

toggleBtn.addEventListener("click", () => {
  panel.classList.toggle("hidden");
  if (!panel.classList.contains("hidden")) input.focus();
});

closeBtn.addEventListener("click", () => {
  panel.classList.add("hidden");
});

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
  } catch (err) {
    messages.removeChild(messages.lastChild);
    addMessage("Network error talking to the assistant.", "bot");
  }
});
