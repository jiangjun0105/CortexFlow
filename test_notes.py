import json

from notes import announce, label, reassure

demo = json.load(open("fixtures/demo.json"))
assert label({"name": "Shakshuka", "description": "x. y."}) == "Shakshuka"
assert "French toast" in reassure("video", "French toast") and "Easy French Toast Recipe" in announce(demo["video"])
assert all(announce(v).startswith("[Status:") for v in (demo["video"], demo["steps"], {"view": "dishes", "meals": demo["meals"]}))
print("ok")
