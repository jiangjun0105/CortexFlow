"""Router checks: python test_router.py (live part needs TYPESAFE_API_KEY)."""
import asyncio
import json
import statistics
import time

import router
from router import _label, pick

S = lambda **kw: {"screen": "welcome", "meals": [], "video": None, "steps": [], "step": 0, "transcript": [], **kw}
meals = [{"id": i, "name": None, "image": "", "description": d} for i, d in enumerate(
    ["Shakshuka is eggs in tomato. Very good.", "A puffy Dutch Baby pancake! Bakes in one pan.", "Avocado toast, simple."])]
vid = {"main": {"id": "x"}, "alternates": []}

assert _label(meals[0]) == "Shakshuka is eggs in tomato." and _label({"name": "Dutch Baby", "description": "x"}) == "Dutch Baby"
assert pick({"recommend": .9}, "what's popular", S()) == ("recommend", None)
assert pick({"recommend": .3}, "thanks!", S()) == ("chat", None)               # nothing >= .5
assert pick({"video": .9, "dish_1": .8}, "the second one", S(screen="dishes", meals=meals)) == ("video", 1)
assert pick({"video": .9, "dish_1": .4}, "make pancakes", S(screen="dishes", meals=meals)) == ("video", None)
assert pick({"steps": .9}, "hard to follow", S()) == ("chat", None)            # guard: no video
assert pick({"steps": .9}, "hard to follow", S(video=vid)) == ("steps", None)
assert pick({"next": .9}, "next", S()) == ("chat", None)                       # guard: no steps
assert pick({"next": .9}, "next", S(steps=[{}] * 6, video=vid)) == ("next", None)
assert pick({}, "go to step four", S(steps=[{}] * 6, video=vid)) == ("goto", 3) # regex, 0-based
assert pick({"next": .9}, "step 2 please", S(steps=[{}] * 6, video=vid)) == ("goto", 1)
assert pick({}, "go to step four", S()) == ("chat", None)                      # no steps loaded
q = router._questions(S(screen="dishes", meals=meals))
assert set(q) == set(router.ROUTES) | {"dish_0", "dish_1", "dish_2"}
print("offline ok")

if not router.KEY:
    print("SKIP live: no TYPESAFE_API_KEY")
    raise SystemExit(0)

demo = json.load(open("fixtures/demo.json"))
fx_meals, fx_video = demo["meals"], {"main": demo["video"]["main"], "alternates": demo["video"]["alternates"]}
fx_steps = demo["steps"]["steps"]
on_video = S(screen="video", meals=fx_meals, video=fx_video,
             transcript=[("user", "the french toast"), ("assistant", "Here's a video for French toast.")])
on_steps = S(screen="steps", meals=fx_meals, video=fx_video, steps=fx_steps, step=1,
             transcript=[("user", "that's hard to follow"), ("assistant", "Here are the steps.")])
CASES = [
    ("I want to cook breakfast for my wife, what's popular lately?", S(), ("recommend", None)),
    ("let's do the shakshuka", S(screen="dishes", meals=fx_meals), ("video", 2)),
    ("the second one looks good", S(screen="dishes", meals=fx_meals), ("video", 1)),
    ("that's hard to follow", on_video, ("steps", None)),
    ("okay next", on_steps, ("next", None)),
    ("wait go back", on_steps, ("back", None)),
    ("say that again", on_steps, ("repeat", None)),
    ("show me that part", on_steps, ("seek", None)),
    ("thanks, she's going to love it", on_steps, ("chat", None)),
    ("pause it", on_video, ("pause", None)),
]


async def live():
    ok, lat = 0, []
    for text, sess, want in CASES:
        t = time.perf_counter()
        r, target, scores = await router.route(text, sess)
        lat.append((time.perf_counter() - t) * 1000)
        ok += (r, target) == want
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"{'ok ' if (r, target) == want else 'BAD'} {text!r} -> {r},{target} want {want} {lat[-1]:.0f}ms top={top}")
        await asyncio.sleep(0.3)
    lat.sort()
    print(f"accuracy {ok}/{len(CASES)}  p50 {statistics.median(lat):.0f}ms  p95 {lat[int(0.95 * (len(lat) - 1))]:.0f}ms")


asyncio.run(live())
