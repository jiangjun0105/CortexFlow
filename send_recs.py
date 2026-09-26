#!/usr/bin/env python3
"""Query RawTree for top-3 breakfast trends, POST them to your backend. Stdlib only.
Usage: RAWTREE_API_KEY=rt_xxx BACKEND_URL=https://your-api/recommendations python3 send_recs.py"""
import json, os, urllib.request
from datetime import datetime, timezone

RT_KEY = os.environ["RAWTREE_API_KEY"]
BACKEND = os.environ["BACKEND_URL"]

sql = """SELECT trend, count() AS posts,
  sum(likes + shares + comments) AS engagement,
  round(avg(likes), 1) AS avg_likes
FROM breakfast_trends_full
WHERE sentiment != 'negative'
GROUP BY trend ORDER BY engagement DESC LIMIT 3"""

req = urllib.request.Request(
    "https://api.rawtree.com/v1/query",
    data=json.dumps({"sql": sql}).encode(),
    headers={"Authorization": f"Bearer {RT_KEY}", "Content-Type": "application/json"})
rows = json.load(urllib.request.urlopen(req, timeout=60))["data"]

payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "recommendations": rows}
req2 = urllib.request.Request(
    BACKEND, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
print("backend ->", urllib.request.urlopen(req2, timeout=30).status)
print(json.dumps(payload, indent=2)[:1500])
