"""Model recommendation and selection regressions for Chinese lecture audio."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asr_recommendations_environment", ROOT / "environment.py")
assert spec and spec.loader
environment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = environment
spec.loader.exec_module(environment)


class ModelRecommendationTests(unittest.TestCase):
    def test_prioritizes_quality_models_without_removing_existing_choices(self):
        candidates = ["tiny", "base", "small", "turbo", "large-v3", "medium", "tiny.en"]
        displayed = environment.order_asr_models(candidates)
        self.assertEqual(displayed[:3], ("turbo", "large-v3", "medium"))
        self.assertEqual(set(displayed), set(candidates))

    def test_prefers_explicit_turbo_name_and_keeps_alias(self):
        candidates = ["tiny", "turbo", "medium", "large-v3-turbo", "large-v3", "small"]
        displayed = environment.order_asr_models(candidates)
        self.assertEqual(displayed[:3], ("large-v3-turbo", "large-v3", "medium"))
        self.assertIn("turbo", displayed)
        self.assertEqual(len(displayed), len(set(displayed)))

    def test_guidance_highlights_quality_and_warns_about_small_models(self):
        self.assertEqual(environment.asr_model_guidance("large-v3-turbo")["badge"], "首选推荐")
        self.assertEqual(environment.asr_model_guidance("large-v3")["badge"], "准确率优先")
        self.assertEqual(environment.asr_model_guidance("medium")["badge"], "资源折中")
        for name in ("tiny", "base", "small", "tiny.en", "medium.en", "distil-small.en"):
            self.assertEqual(environment.asr_model_guidance(name)["tone"], "caution")
        self.assertEqual(environment.asr_model_guidance("turbo")["tone"], "alias")
        self.assertEqual(environment.asr_model_guidance("distil-large-v3"), {})

    def test_supported_model_source_is_respected(self):
        fake_whisper = types.ModuleType("faster_whisper")
        fake_utils = types.ModuleType("faster_whisper.utils")
        fake_utils._MODELS = {"small": "small", "large-v3": "large", "turbo": "turbo"}
        fake_whisper.utils = fake_utils
        with patch.dict(sys.modules, {"faster_whisper": fake_whisper, "faster_whisper.utils": fake_utils}):
            result = environment.available_asr_models()
        self.assertEqual(result, ("turbo", "large-v3", "small"))

    def test_template_shows_guidance_without_changing_download_actions(self):
        template = (ROOT / "templates" / "transcription_settings.html").read_text(encoding="utf-8")
        self.assertIn('model_guidance.get(item.model, {})', template)
        self.assertIn('guidance.get("badge")', template)
        self.assertIn('guidance.get("description")', template)
        self.assertIn("data-asr-model-choice", template)
        self.assertIn("data-asr-model-action", template)
        self.assertIn('name="asr_model"', template)


if __name__ == "__main__":
    unittest.main()
