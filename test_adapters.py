import pytest

from recommender import label
from search_agent import FIXTURE, adapt_video, youtube_id


def test_adapt_video_fixture():
    assert adapt_video(FIXTURE["raw"]["video"], "French toast") == FIXTURE["video"]


def test_youtube_url_forms():
    for url in ["https://www.youtube.com/watch?v=abc123&t=5", "https://youtu.be/abc123?si=x",
                "https://youtube.com/shorts/abc123", "https://www.youtube.com/embed/abc123"]:
        assert youtube_id(url) == "abc123", url


def test_non_youtube_raises():
    with pytest.raises(ValueError):
        youtube_id("https://vimeo.com/12345")


def test_title_fallback_first_sentence():
    v = adapt_video({"video_url": "https://youtu.be/x1", "description": "Crispy toast. Very good."}, "d")
    assert v["main"]["title"] == "Crispy toast."


def test_label():
    assert label({"name": "Shakshuka", "description": "Eggs. In sauce."}) == "Shakshuka"
    assert label({"name": None, "description": "Eggs in sauce! Yum."}) == "Eggs in sauce!"
