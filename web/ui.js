// Page shell: two-column layout, orb avatar. Sets window.UI. Load after views.js.
(() => {
  let video = null;
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
    left.append(orb);
    for (const id of ["caption", "talk", "status"]) { const e = document.getElementById(id); if (e) left.append(e); }
    main.append(left, stage);
    document.body.prepend(main);
    document.body.dataset.state ||= "idle";
    rate(document.body.dataset.state);
    Views.mount(stage);
    Views.onAction = (a) => UI.onAction(a);
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
      } catch (e) { console.error("UI.handle", e); }
    },
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
