import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.http.emotion_routes import router as emotion_router
from app.domain.entities import EmotionPrediction, EmotionResult


class FakeDetectEmotionUseCase:
    def execute(self, text: str, top_k: int = 3) -> EmotionResult:
        _ = top_k
        return EmotionResult(
            text=text,
            label="joy",
            score=0.91,
            predictions=[
                EmotionPrediction(label="joy", score=0.91),
                EmotionPrediction(label="surprise", score=0.06),
            ],
        )


class EmotionRouteTests(unittest.TestCase):
    def test_detect_emotion_success(self) -> None:
        app = FastAPI()
        app.state.emotion_use_case = FakeDetectEmotionUseCase()
        app.include_router(emotion_router)

        client = TestClient(app)
        response = client.post("/emotion/detect", json={"text": "I feel happy", "top_k": 2})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "joy")
        self.assertTrue(len(payload["predictions"]) >= 1)

    def test_detect_emotion_requires_text(self) -> None:
        app = FastAPI()
        app.state.emotion_use_case = FakeDetectEmotionUseCase()
        app.include_router(emotion_router)

        client = TestClient(app)
        response = client.post("/emotion/detect", json={"text": "   "})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
