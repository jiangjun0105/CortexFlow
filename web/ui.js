// Page shell: two-column layout, orb avatar, metrics strip. Sets window.UI. Load after views.js.
(() => {
  let strip = null, video = null;
  const rate = (s) => { if (video) video.playbackRate = video.defaultPlaybackRate = s === "speaking" ? 1 : s === "idle" ? 0.6 : 0.85; };

  function build() {
    const main = document.createElement("main"); main.className = "ui";
    const left = document.createElement("aside"); left.className = "ui-left";
    const orb = document.createElement("div"); orb.className = "avatar-wrap";
    video = document.createElement("video"); video.className = "avatar";
    Object.assign(video, { src: "avatar.mp4", autoplay: true, muted: true, loop: true, playsInline: true });
    video.onerror = () => { video.remove(); video = null; orb.className = "orb"; }; // fallback: old CSS orb
    orb.append(video);
    const stage = document.getElementById("stage") || document.createElement("section");
    stage.id = "stage"; stage.classList.add("ui-stage");
    // ponytail: own strip element, because app.js overwrites #metrics.textContent
    strip = document.createElement("div"); strip.className = "ui-metrics";
    left.append(orb);
    for (const id of ["caption", "talk", "status"]) { const e = document.getElementById(id); if (e) left.append(e); }
    left.append(strip);
    document.body.classList.add("no-metrics"); // hidden by default; M shows the timings for debugging
    const m = document.getElementById("metrics"); if (m) left.append(m);
    main.append(left, stage);
    document.body.prepend(main);
    document.body.dataset.state ||= "idle";
    rate(document.body.dataset.state);
    Views.mount(stage);
    Views.onAction = (a) => UI.onAction(a);
    document.addEventListener("keydown", (e) => {
      if ((e.key === "m" || e.key === "M") && !e.target.closest?.("input,textarea,[contenteditable]"))
        document.body.classList.toggle("no-metrics");
    });
  }

  function renderMetrics(msg) {
    strip.replaceChildren();
    const route = document.createElement("div"); route.className = "ui-route";
    const target = [msg.target, msg.dish, msg.index].find((v) => v != null);
    route.textContent = `→ ${msg.route ?? "–"}${target != null ? " · " + target : ""}`;
    strip.append(route);
    const ms = Object.entries(msg.ms || {}).filter(([, v]) => typeof v === "number");
    const max = Math.max(1, ...ms.map(([, v]) => v));
    for (const [k, v] of ms) {
      const row = document.createElement("div"); row.className = "ui-bar";
      const lab = document.createElement("span"); lab.textContent = k;
      const bar = document.createElement("i"); bar.style.width = `${(v / max) * 100}%`;
      const val = document.createElement("b"); val.textContent = `${Math.round(v)}`;
      row.append(lab, bar, val); strip.append(row);
    }
  }

  window.UI = {
    onAction: () => {},
    level(x) { document.body.style.setProperty("--level", Math.max(0, Math.min(1, +x || 0))); },
    handle(msg) {
      try {
        if (!msg || typeof msg !== "object") return;
        if (msg.type === "state") {
          const s = msg.value ?? msg.state; // server protocol uses "value"
          document.body.dataset.state = s; rate(s);
          if (["listening", "hearing", "speaking"].includes(s)) Views.pauseVideo();
        } else if (msg.type === "view") Views.show(msg);
        else if (msg.type === "control" && msg.name !== "interrupt") Views.control(msg);
        else if (msg.type === "metrics" && strip) renderMetrics(msg);
      } catch (e) { console.error("UI.handle", e); }
    },
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
