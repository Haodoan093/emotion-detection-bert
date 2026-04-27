import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.application.use_cases import GenerateFinancialAnswerUseCase
from app.infrastructure.gemini_llm import GeminiLLMService
from app.infrastructure.local_mistral import LocalMistralService


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM CLI")
    parser.add_argument("--provider", choices=["gemini", "local"], default="gemini")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--gemini-key", default="")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash")
    parser.add_argument("--local-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.provider == "gemini":
        if not args.gemini_key:
            raise SystemExit("--gemini-key is required for Gemini provider")
        service = GeminiLLMService(args.gemini_key, args.gemini_model)
    else:
        service = LocalMistralService(args.local_model, args.max_new_tokens)

    use_case = GenerateFinancialAnswerUseCase(service)
    result = use_case.execute(question=args.question, context=args.context)
    print(result.answer)


if __name__ == "__main__":
    main()

