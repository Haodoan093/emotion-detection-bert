import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.infrastructure.whisper_asr import WhisperASRService


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper ASR CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 decoding (requires CUDA)")
    parser.add_argument("--no-fp16", action="store_true", help="Force fp32 decoding")
    parser.add_argument("--language", default="vi", help="Transcription language (default: vi)")
    parser.add_argument("--initial-prompt", default=None, help="Optional domain prompt to bias decoding")
    args = parser.parse_args()

    fp16: bool | None = None
    if args.fp16:
        fp16 = True
    if args.no_fp16:
        fp16 = False

    service = WhisperASRService(
        model_name=args.model,
        device=args.device,
        use_fp16=fp16,
        language=args.language,
        initial_prompt=args.initial_prompt,
    )
    service.load_model()
    text = service.transcribe(args.audio)
    print(text)


if __name__ == "__main__":
    main()

