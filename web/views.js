// Stage views: welcome | dishes | video | steps. Sets window.Views (see plan "Frontend split").
(() => {
  const CHIPS = ["What's popular for breakfast?", "How do I make French toast?", "Show me the steps"];
  let root = null, current = null;
  const panes = {};                                       // view name -> section element
  const data = { dishes: null, video: null, steps: null };
  let stepIndex = 0;

  // video state
  let player = null, playerReady = false, queue = [];     // queued player ops until onReady
  let playlist = [], playIndex = 0, savedPos = 0;

  const emit = (a) => { try { Views.onAction(a); } catch (e) { console.error(e); } };

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;              // textContent: web data is untrusted
    return e;
  }

  function activate(name) {
    if (current === name) return;
    if (current === "video") { savedPos = playerTime(); playerDo((p) => p.pauseVideo()); }
    for (const [k, p] of Object.entries(panes)) {
      p.classList.toggle("active", k === name);
      p.classList.toggle("left", k !== name && name === "steps" && k === "video");
    }
    current = name;
    root.dataset.view = name;
  }

  // ---------- welcome ----------
  function renderWelcome() {
    const p = panes.welcome; p.replaceChildren();
    p.append(el("h1", "vw-greet", "Good morning — what are we cooking?"));
    const row = el("div", "vw-chips");
    for (const t of CHIPS) {
      const c = el("button", "vw-chip", t);
      c.onclick = () => emit({ type: "text", text: t });
      row.append(c);
    }
    p.append(row);
  }

  // ---------- dishes ----------
  function renderDishes() {
    const p = panes.dishes; p.replaceChildren();
    p.append(el("h2", "vw-title", "Trending breakfasts"));
    const grid = el("div", "vw-cards");
    (data.dishes.meals || []).forEach((m, i) => {
      const card = el("button", "vw-card");
      card.style.animationDelay = `${i * 90}ms`;
      const img = el("div", "vw-img");
      if (m.image) {
        const im = el("img"); im.src = m.image; im.alt = m.name || ""; im.loading = "lazy";
        im.onerror = () => im.remove();                  // gradient on .vw-img shows through
        img.append(im);
      }
      card.append(img);
      const body = el("div", "vw-card-body");
      if (m.name) body.append(el("h3", "vw-card-name", m.name));
      if (m.minutes != null) body.append(el("div", "vw-card-meta", `${m.minutes} min`));
      if (m.description) body.append(el("p", "vw-card-desc", m.description));
      card.append(body);
      card.onclick = () => { selectDish(m.id); emit({ type: "action", name: "select_dish", id: m.id }); };
      card.dataset.id = m.id;
      grid.append(card);
    });
    p.append(grid, el("p", "vw-hint", 'Say "the first one", or tap a card'));
  }

  function selectDish(id) {
    for (const c of panes.dishes.querySelectorAll(".vw-card")) {
      const sel = c.dataset.id === String(id);
      c.classList.toggle("selected", sel);
      c.classList.toggle("faded", !sel);
    }
  }

  // ---------- video ----------
  function renderVideo() {
    const p = panes.video, v = data.video; p.replaceChildren();
    p.append(el("h2", "vw-title", v.dish || v.main?.title || ""));
    const frame = el("div", "vw-player");
    frame.append(el("div", null)); frame.firstChild.id = "vw-yt";
    p.append(frame);
    const alts = v.alternates || [];
    if (alts.length) {
      p.append(el("div", "vw-sub", "Other videos"));
      const list = el("div", "vw-alts");
      alts.forEach((a) => {
        const row = el("button", "vw-alt");
        const th = el("img"); th.src = a.thumb || `https://i.ytimg.com/vi/${a.id}/hqdefault.jpg`; th.alt = "";
        row.append(th, el("span", null, a.title + (a.minutes ? ` · ${a.minutes} min` : "")));
        row.onclick = () => { playVideoId(a.id); emit({ type: "action", name: "select_video", id: a.id }); };
        list.append(row);
      });
      p.append(list);
    }
    playlist = [v.main, ...alts].filter(Boolean);
    playIndex = 0; savedPos = 0;
    player = null; playerReady = false; queue = [];
    if (playlist.length) createPlayer(playlist[0].id);
  }

  function createPlayer(id) {
    if (!(window.YT && YT.Player)) {                      // API not loaded yet: poll (host may own onYouTubeIframeAPIReady)
      clearTimeout(createPlayer.t);
      createPlayer.t = setTimeout(() => { if (!player) createPlayer(playlist[playIndex].id); }, 200);
      return;
    }
    if (!document.getElementById("vw-yt")) return;
    player = new YT.Player("vw-yt", {
      videoId: id,
      playerVars: { autoplay: 0, playsinline: 1, rel: 0, modestbranding: 1 }, // never auto-start: it drowns out the agent
      events: {
        onReady: (e) => {
          playerReady = true;
          queue.splice(0).forEach((f) => f(e.target));
        },
        onError: onPlayerError,
      },
    });
  }

  function onPlayerError() {
    if (playIndex + 1 < playlist.length) { playIndex++; playVideoId(playlist[playIndex].id); return; }
    const v = playlist[playIndex] || {};
    const frame = panes.video.querySelector(".vw-player");
    const fb = el("a", "vw-fallback");
    fb.href = `https://www.youtube.com/watch?v=${v.id}`; fb.target = "_blank"; fb.rel = "noopener";
    const th = el("img"); th.src = v.thumb || `https://i.ytimg.com/vi/${v.id}/hqdefault.jpg`; th.alt = "";
    fb.append(th, el("span", null, "▶ Open on YouTube"));
    try { player && player.destroy(); } catch {}
    player = null; playerReady = false;
    frame.replaceChildren(fb);
  }

  // seekTo starts a cued/unstarted video (YouTube API), so re-pause unless it was already playing
  function seekQuiet(p, t) { const playing = p.getPlayerState() === 1; p.seekTo(t, true); if (!playing) p.pauseVideo(); }

  function playVideoId(id) {
    const i = playlist.findIndex((x) => x.id === id);
    if (i >= 0) playIndex = i;
    savedPos = 0;
    if (player) playerDo((p) => p.cueVideoById(id)); // cue, don't play
    else { panes.video.querySelector(".vw-player").replaceChildren(Object.assign(el("div"), { id: "vw-yt" })); createPlayer(id); }
  }

  function playerDo(f) { if (player && playerReady) f(player); else if (player) queue.push(f); }
  function playerTime() { try { return player && playerReady ? player.getCurrentTime() : savedPos; } catch { return savedPos; } }

  // ---------- steps ----------
  function renderSteps() {
    const p = panes.steps, s = data.steps, steps = s.steps || [], n = steps.length;
    p.replaceChildren();
    if (!n) { p.append(el("h2", "vw-title", "No steps yet")); return; }
    stepIndex = Math.max(0, Math.min(stepIndex, n - 1));
    const st = steps[stepIndex];
    const head = el("div", "vw-steps-head");
    head.append(el("h2", "vw-title", s.dish || ""));
    const prog = el("div", "vw-progress");
    prog.append(el("span", "vw-count", `Step ${stepIndex + 1} of ${n}`));
    const dots = el("span", "vw-dots");
    steps.forEach((_, i) => dots.append(el("i", i === stepIndex ? "on" : i < stepIndex ? "done" : "")));
    prog.append(dots); head.append(prog);

    const card = el("div", "vw-step");
    card.append(el("div", "vw-step-icon", st.icon || "🍳"), el("h3", "vw-step-title", st.title));
    const tags = el("div", "vw-tags");
    (st.tags || []).forEach((t) => tags.append(el("span", "vw-tag", t)));
    card.append(tags, el("p", "vw-step-detail", st.detail || ""));

    const nav = el("div", "vw-nav");
    const prev = el("button", "vw-arrow", stepIndex > 0 ? `← ${steps[stepIndex - 1].title}` : "←");
    const next = el("button", "vw-arrow", stepIndex < n - 1 ? `${steps[stepIndex + 1].title} →` : "→");
    prev.disabled = stepIndex === 0; next.disabled = stepIndex === n - 1;
    prev.onclick = () => userStep(stepIndex - 1);
    next.onclick = () => userStep(stepIndex + 1);
    nav.append(prev, next);

    const back = el("button", "vw-chip vw-back", "▶ back to video");
    back.onclick = () => {
      emit({ type: "action", name: "show_video" });
      if (data.video) { activate("video"); resumeVideo(); }
    };
    p.append(head, card, nav, back);
  }

  function gotoStep(i, dir) {
    const n = data.steps?.steps?.length || 0;
    if (!n || i < 0 || i >= n) return false;
    stepIndex = i;
    renderSteps();
    const card = panes.steps.querySelector(".vw-step");
    if (card && dir) card.classList.add(dir > 0 ? "from-right" : "from-left");
    return true;
  }
  function userStep(i) {
    const d = i - stepIndex;
    if (gotoStep(i, d)) emit({ type: "action", name: "goto_step", index: i });
  }

  function resumeVideo() {
    const pos = savedPos;
    playerDo((p) => { if (pos) seekQuiet(p, pos); });
  }

  // ---------- public ----------
  window.Views = {
    onAction: () => {},

    mount(stageEl) {
      root = stageEl;
      root.classList.add("vw-stage");
      for (const k of ["welcome", "dishes", "video", "steps"]) {
        panes[k] = el("section", `vw-pane vw-${k}`);
        root.append(panes[k]);
      }
      renderWelcome();
      activate("welcome");
      document.addEventListener("keydown", (e) => {
        if (current !== "steps" || e.target.closest?.("input,textarea,[contenteditable]")) return;
        if (e.key === "ArrowRight") userStep(stepIndex + 1);
        else if (e.key === "ArrowLeft") userStep(stepIndex - 1);
      });
    },

    show(msg) {
      const name = msg.view;
      if (!panes[name]) return console.warn("Views: unknown view", name);
      if (name === "welcome") { activate("welcome"); return; }
      const same = JSON.stringify(data[name]) === JSON.stringify(msg);
      if (name === "video") {
        // Same video payload (e.g. back from steps): keep the player, resume where we left off.
        const firstShow = !same || !player;
        if (!same) { data.video = msg; renderVideo(); }
        activate("video");
        if (!firstShow) resumeVideo();
        return;
      }
      if (!same) {
        data[name] = msg;
        if (name === "steps") { stepIndex = 0; renderSteps(); }
        else renderDishes();
      }
      activate(name);
      if (name === "steps") panes.steps.querySelector(".vw-step")?.classList.add("deal");
    },

    control(msg) {
      if (msg.name === "goto_step") { gotoStep(msg.index, msg.index - stepIndex); return; }
      if (msg.name !== "video") return;
      if (msg.cmd === "play") playerDo((p) => p.playVideo());
      else if (msg.cmd === "pause") this.pauseVideo();
      else if (msg.cmd === "seek") { savedPos = msg.t || 0; playerDo((p) => seekQuiet(p, msg.t || 0)); }
    },

    pauseVideo() { playerDo((p) => p.pauseVideo()); },
    duck(on) { playerDo((p) => p.setVolume(on ? 15 : 100)); }, // quieter while the agent talks
  };
})();
