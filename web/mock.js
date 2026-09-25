// Backend-free replay of the breakfast conversation (spec §1) through window.UI. Keys: N next, A autoplay, R restart.
(async () => {
  const d = await (await fetch("../fixtures/demo.json")).json();
  const names = ["French toast", "Miso soup", "Shakshuka"];
  const meals = d.meals.map((m, i) => ({ ...m, name: m.name || names[i] }));
  const $ = (id) => document.getElementById(id);
  const UI = window.UI || { handle: (m) => console.log("no UI.handle", m), level() {} };
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  let levelTimer = null;
  function send(msg) {
    if (msg.type === "caption") $("caption").textContent += ($("caption").textContent ? " " : "") + msg.text;
    if (msg.type === "state") {
      clearInterval(levelTimer); levelTimer = null;
      if (msg.value === "speaking") levelTimer = setInterval(() => UI.level(Math.random()), 80);
      else UI.level(0);
    }
    try { UI.handle(msg); } catch (e) { console.error(e); }
  }
  const metrics = (route, target) => ({ type: "metrics", route, target,
    ms: { asr: 310, jev: 290, voice_first_audio: 820, module: 2100, total: 4200 } });

  // One agent turn: user line, listening → thinking → caption → speaking → extra msgs → metrics → idle.
  const turn = (you, caption, msgs, route) => async () => {
    $("caption").textContent = "";
    $("status").textContent = "you: " + you;
    send({ type: "state", value: "listening" }); await wait(700);
    send({ type: "state", value: "thinking" }); await wait(600);
    send({ type: "caption", text: caption });
    send({ type: "state", value: "speaking" }); await wait(900);
    for (const m of msgs) { send(m); await wait(500); }
    await wait(800);
    send(metrics(route, 0));
    send({ type: "state", value: "idle" });
  };

  const script = [
    turn("what's popular for breakfast?", "Sure, let me see what's trending…", [{ type: "view", view: "dishes", meals }], "dishes"),
    turn("let's do the French toast", "Great pick, finding a video.", [{ type: "view", ...d.video }], "video"),
    turn("that's hard to follow", "No problem, here it is step by step.", [{ type: "view", ...d.steps }], "steps"),
    turn("next", "Next up, heat the pan.", [{ type: "control", name: "goto_step", index: 1 }], "control"),
    turn("next", "Now dip the bread.", [{ type: "control", name: "goto_step", index: 2 }], "control"),
    turn("show me this part in the video", "Here's that part.",
      [{ type: "view", ...d.video }, { type: "control", name: "video", cmd: "seek", t: 60 }], "control"),
  ];

  let i = 0, busy = false, auto = false;
  async function next() {
    if (busy || i >= script.length) return;
    busy = true;
    try { await script[i++](); } finally { busy = false; }
    if (auto && i < script.length) setTimeout(next, 3000);
  }
  function restart() {
    i = 0; auto = false;
    $("caption").textContent = ""; $("status").textContent = "mock"; $("metrics").textContent = "";
    send({ type: "state", value: "idle" });
    send({ type: "view", view: "welcome" });
  }

  UI.onAction = (a) => {
    console.log("UI.onAction", a);
    if (a.name === "select_dish" || a.type === "select_dish") next();
  };
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === "n") next();
    else if (k === "a") { auto = !auto; if (auto) next(); }
    else if (k === "r") restart();
  });
  restart();
})();
