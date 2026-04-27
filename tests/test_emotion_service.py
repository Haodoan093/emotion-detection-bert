import unittest

from app.infrastructure.hf_emotion import HfEmotionService


class EmotionServiceUnitTests(unittest.TestCase):
    def test_normalize_predictions_sorts_descending(self) -> None:
        raw = [
            {"label": "joy", "score": 0.21},
            {"label": "anger", "score": 0.62},
            {"label": "sadness", "score": 0.17},
        ]
        predictions = HfEmotionService._normalize_predictions(raw)
        self.assertEqual(predictions[0].label, "anger")
        self.assertAlmostEqual(predictions[0].score, 0.62)

    def test_resolve_device_cpu(self) -> None:
        self.assertEqual(HfEmotionService._resolve_device("cpu"), -1)


if __name__ == "__main__":
    unittest.main()
