import json

from notes import announce, label

demo = json.load(open("fixtures/demo.json"))
dishes = {"view": "dishes", "meals": demo["meals"]}

assert label({"name": "Shakshuka", "description": "x. y."}) == "Shakshuka"
assert label(demo["meals"][0]) == "A small scoop of flour makes this the best French toast recipe!"

a = announce(dishes)
assert a.startswith("[The screen now shows: 3 breakfasts: A small scoop") and a.count("; ") == 2, a
assert "Japanese Miso Soup!" in a and "To make shakshuka" in a and a.endswith("continue.]")

assert "a video: Easy French Toast Recipe." in announce(demo["video"])
assert "a 8-minute video: X." in announce({"view": "video", "main": {"title": "X", "minutes": 8}})
assert "6 steps; step 1 is Whisk the batter." in announce(demo["steps"])
print("ok")
