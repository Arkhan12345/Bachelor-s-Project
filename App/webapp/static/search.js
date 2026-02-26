(() => {
  const geneInput = document.getElementById("gene");
  const dropdown = document.getElementById("gene_dropdown");
  const entrez = document.getElementById("entrez");
  const symbol = document.getElementById("symbol");
  const genetitle = document.getElementById("genetitle");

  if (!geneInput || !dropdown || !entrez || !symbol || !genetitle) return;

  const wrapper = geneInput.closest(".gene-field") || geneInput.parentElement;

  let lastItems = [];
  let activeIndex = -1;
  let timer = null;
  let lastQuery = "";

  function close() {
    dropdown.classList.add("hidden");
    dropdown.innerHTML = "";
    activeIndex = -1;
  }

  function open() {
    if (lastItems.length) dropdown.classList.remove("hidden");
  }

  function render(items) {
    lastItems = items || [];
    activeIndex = -1;

    if (!lastItems.length) {
      close();
      return;
    }

    dropdown.innerHTML = lastItems.map((it, i) => {
      const meta = `${it.symbol || ""}${it.entrez ? " (" + it.entrez + ")" : ""}`;
      return `
        <div class="suggestion-item" role="option" data-idx="${i}">
          <div>${it.label}</div>
          <span class="suggestion-meta">${meta}</span>
        </div>
      `;
    }).join("");

    open();
  }

  function setActive(i) {
    const els = Array.from(dropdown.querySelectorAll(".suggestion-item"));
    els.forEach(el => el.classList.remove("active"));
    if (i < 0 || i >= els.length) { activeIndex = -1; return; }
    els[i].classList.add("active");
    els[i].scrollIntoView({ block: "nearest" });
    activeIndex = i;
  }

  function choose(it) {
    if (!it) return;
    geneInput.value = it.label;
    entrez.value = it.entrez || "";
    symbol.value = it.symbol || "";
    genetitle.value = it.genetitle || "";
    close();
  }

  async function fetchSuggest(q) {
    const res = await fetch(`/api/gene_suggest?q=${encodeURIComponent(q)}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.items || [];
  }

  geneInput.addEventListener("input", () => {
    const q = geneInput.value.trim();

    entrez.value = "";
    symbol.value = "";
    genetitle.value = "";

    lastQuery = q;
    clearTimeout(timer);

    if (q.length < 1) { close(); return; }

    timer = setTimeout(async () => {
      try {
        const items = await fetchSuggest(q);
        if (lastQuery !== q) return;
        render(items);
      } catch {
        close();
      }
    }, 150);
  });

  geneInput.addEventListener("focus", () => {
    if (lastItems.length) open();
  });

  geneInput.addEventListener("keydown", (e) => {
    if (dropdown.classList.contains("hidden")) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive(Math.min(activeIndex + 1, lastItems.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive(Math.max(activeIndex - 1, 0));
    } else if (e.key === "Enter") {
      if (activeIndex >= 0) {
        e.preventDefault();
        choose(lastItems[activeIndex]);
      }
    } else if (e.key === "Escape") {
      close();
    }
  });

  // Use mousedown so selection happens before focus changes
  dropdown.addEventListener("mousedown", (e) => {
    const el = e.target.closest(".suggestion-item");
    if (!el) return;
    const idx = parseInt(el.getAttribute("data-idx"), 10);
    if (!Number.isNaN(idx)) choose(lastItems[idx]);
  });

  // Close only when clicking OUTSIDE the wrapper (robust)
  document.addEventListener("mousedown", (e) => {
    if (wrapper && wrapper.contains(e.target)) return;
    close();
  });
})();