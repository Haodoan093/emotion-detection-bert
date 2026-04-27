import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.infrastructure.hf_emotion import HfEmotionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion detection CLI")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--model", default="j-hartmann/emotion-english-distilroberta-base")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    service = HfEmotionService(model_name=args.model, device=args.device)
    result = service.detect(text=args.text, top_k=args.top_k)
    print(f"label={result.label} score={result.score:.4f}")
    for item in result.predictions:
        print(f"- {item.label}: {item.score:.4f}")


if __name__ == "__main__":
    main()

