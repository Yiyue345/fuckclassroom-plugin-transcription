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
    def test_prioritizes_quality_models_without_showing_duplicate_aliases(self):
        candidates = [
            "tiny", "base", "small", "turbo", "large", "large-v3",
            "medium", "large-v3-turbo", "tiny.en",
        ]
        displayed = environment.order_asr_models(candidates)
        self.assertEqual(displayed[:3], ("large-v3-turbo", "large-v3", "medium"))
        self.assertNotIn("turbo", displayed)
        self.assertNotIn("large", displayed)
        self.assertEqual(len(displayed), len(set(displayed)))
        self.assertIn("tiny.en", displayed)

    def test_only_alias_supported_still_remains_selectable(self):
        displayed = environment.order_asr_models(["tiny", "turbo", "medium"])
        self.assertEqual(displayed, ("turbo", "medium", "tiny"))

    def test_deduplicates_any_shared_repository_not_only_known_aliases(self):
        repos = {
            "large-v3": "org/large-v3",
            "large": "org/large-v3",
            "large-v3-turbo": "org/turbo",
            "turbo": "org/turbo",
            "surprise-alias": "org/medium",
            "medium": "org/medium",
            "tiny": "org/tiny",
        }
        displayed = environment.order_asr_models(list(repos), repos)
        self.assertEqual(displayed, ("large-v3-turbo", "large-v3", "medium", "tiny"))

    def test_guidance_highlights_quality_and_warns_about_small_models(self):
        self.assertEqual(environment.asr_model_guidance("large-v3-turbo")["badge"], "首选推荐")
        self.assertEqual(environment.asr_model_guidance("large-v3")["badge"], "准确率优先")
        self.assertEqual(environment.asr_model_guidance("medium")["badge"], "资源折中")
        for name in ("tiny", "base", "small", "tiny.en", "medium.en", "distil-small.en"):
            self.assertEqual(environment.asr_model_guidance(name)["tone"], "caution")
        self.assertEqual(environment.asr_model_guidance("distil-large-v3"), {})

    def test_supported_model_source_is_respected_and_aliases_normalized(self):
        fake_whisper = types.ModuleType("faster_whisper")
        fake_utils = types.ModuleType("faster_whisper.utils")
        fake_utils._MODELS = {
            "small": "Systran/faster-whisper-small",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "Systran/faster-whisper-large-v3",
            "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        }
        fake_whisper.utils = fake_utils
        with patch.dict(sys.modules, {
            "faster_whisper": fake_whisper, "faster_whisper.utils": fake_utils
        }):
            choices = environment.available_asr_models()
            aliases = environment.asr_model_aliases()
            supported = environment.all_asr_model_names()
        self.assertEqual(choices, ("large-v3-turbo", "large-v3", "small"))
        self.assertEqual(aliases, {"large": "large-v3", "turbo": "large-v3-turbo"})
        self.assertIn("turbo", supported)
        self.assertIn("large", supported)

    def test_template_shows_guidance_without_changing_download_actions(self):
        template = (ROOT / "templates" / "transcription_settings.html").read_text(encoding="utf-8")
        self.assertIn('model_guidance.get(item.model, {})', template)
        self.assertIn('guidance.get("badge")', template)
        self.assertIn('guidance.get("description")', template)
        self.assertIn("data-asr-model-choice", template)
        self.assertIn("data-asr-model-action", template)
        self.assertIn('name="asr_model"', template)
        self.assertIn('value="{{ display_model }}"', template)
        self.assertIn('get(settings.asr_model, settings.asr_model)', template)
        engine = (ROOT / "engine.py").read_text(encoding="utf-8")
        self.assertIn('from .environment import all_asr_model_names', engine)


if __name__ == "__main__":
    unittest.main()
