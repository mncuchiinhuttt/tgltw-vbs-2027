"""Tests for the DRES 2.x submission contract (_build_dres_answers +
parse_submission_verdict + the /api/dres/submit request model).

No network: the DRES call is mocked; the router's duplicate guard and
session gate are exercised where cheap.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "webapp", "backend"))
sys.path.append(REPO_ROOT)

import main as backend  # noqa: E402
from main import DresSubmitRequest, _build_dres_answers  # noqa: E402
import dres_client  # noqa: E402


class TestBuildDresAnswers(unittest.TestCase):
    def test_kis_shot_converts_seconds_to_ms(self):
        r = DresSubmitRequest(task_id="t", payload={"mediaItemName": "v.mp4", "timestamp": 12.3456},
                              video_name="v.mp4")
        self.assertEqual(_build_dres_answers(r),
                         [{"mediaItemName": "v.mp4", "start": 12346, "end": 12346}])

    def test_vqa_text_answer_stripped(self):
        r = DresSubmitRequest(task_id="t", payload=None, answer_text="  a red car ")
        self.assertEqual(_build_dres_answers(r), [{"text": "a red car"}])

    def test_vqa_whitespace_falls_through_to_media(self):
        r = DresSubmitRequest(task_id="t", payload=None, answer_text="   ", video_name="v.mp4")
        answers = _build_dres_answers(r)
        self.assertEqual(answers[0]["mediaItemName"], "v.mp4")

    def test_trake_segments_ordered_ms(self):
        r = DresSubmitRequest(task_id="t", payload=None, segments=[
            {"mediaItemName": "v.mp4", "timestamp": 1.5},
            {"mediaItemName": "v.mp4", "start": 10, "end": 12.25},
        ])
        # segment end extends to the next segment's start (no zero-width ranges)
        self.assertEqual(_build_dres_answers(r), [
            {"mediaItemName": "v.mp4", "start": 1500, "end": 10000},
            {"mediaItemName": "v.mp4", "start": 10000, "end": 12250},
        ])


class TestVerdictParsing(unittest.TestCase):
    def test_status_list(self):
        self.assertEqual(dres_client.parse_submission_verdict(
            {"answers": [{"status": "CORRECT"}, {"status": "WRONG"}]}), "CORRECT, WRONG")

    def test_unknown_fallback(self):
        # no answers field -> falls back to a raw-JSON preview
        self.assertIn("odd", dres_client.parse_submission_verdict({"odd": 1}))


class TestSubmitRoute(unittest.TestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(backend.app)

    def test_vqa_submission_reaches_dres_with_text_answer(self):
        backend._dres_session_id = "sess"
        with patch.object(backend, "_dres_config",
                          return_value={"base_url": "http://dres", "evaluation_id": "e1"}), \
             patch.object(backend.dres_client.requests, "post", return_value=self._ok()) as post, \
             patch.object(backend.interaction_log, "log_interaction"):
            resp = self._client().post("/api/dres/submit", json={
                "task_id": "42", "answer_text": "a red car"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["verdict"], "CORRECT")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body, {"taskId": "42", "answers": [{"text": "a red car"}]})

    def test_trake_submission_sends_ordered_segments(self):
        backend._dres_session_id = "sess"
        with patch.object(backend, "_dres_config",
                          return_value={"base_url": "http://dres", "evaluation_id": "e1"}), \
             patch.object(backend.dres_client.requests, "post", return_value=self._ok()) as post, \
             patch.object(backend.interaction_log, "log_interaction"):
            resp = self._client().post("/api/dres/submit", json={
                "task_id": "42",
                "segments": [{"mediaItemName": "v.mp4", "timestamp": 1.5},
                             {"mediaItemName": "v.mp4", "timestamp": 9.0}]})
        self.assertEqual(resp.status_code, 200)
        # the ordered segments live in the POST body sent to DRES
        answers = post.call_args.kwargs["json"]["answers"]
        self.assertEqual(answers, [
            {"mediaItemName": "v.mp4", "start": 1500, "end": 9000},
            # the last segment closes after a nominal 1s
            {"mediaItemName": "v.mp4", "start": 9000, "end": 10000},
        ])

    def test_dres_rejection_surfaces_body(self):
        backend._dres_session_id = "sess"
        bad = MagicMock(status_code=400, text='{"detail": "task not active"}')
        bad.json.side_effect = ValueError()
        with patch.object(backend, "_dres_config",
                          return_value={"base_url": "http://dres", "evaluation_id": "e1"}), \
             patch.object(backend.dres_client.requests, "post", return_value=bad), \
             patch.object(backend.interaction_log, "log_interaction"):
            resp = self._client().post("/api/dres/submit", json={
                "task_id": "42", "payload": {"mediaItemName": "v.mp4", "timestamp": 1.0}})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("task not active", resp.json()["detail"])

    @staticmethod
    def _ok():
        ok = MagicMock(status_code=200, text="")
        ok.json.return_value = {"answers": [{"status": "CORRECT"}]}
        return ok


if __name__ == "__main__":
    unittest.main()
