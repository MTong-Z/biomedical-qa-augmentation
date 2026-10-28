import argparse
import json

from . import __version__


def model_options(parser, adapter=False, api_teacher=False):
    parser.add_argument("--model", required=True, help="Explicit provider model ID or local model directory")
    parser.add_argument("--revision", required=not api_teacher, help="Exact revision required for Hugging Face models")
    if api_teacher:
        parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
        parser.add_argument("--max-api-calls", type=int, help="Required per-command network call cap for the API teacher; cache hits do not count")
        parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], help="Only set an effort supported by the chosen API model")
    parser.add_argument("--four-bit", action="store_true", help="Load in four bit precision on a compatible CUDA GPU")
    parser.add_argument("--seed", type=int, default=42)
    if adapter:
        parser.add_argument("--adapter", help="Optional trained LoRA adapter directory")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)


def generator(args):
    from .model import HFGenerator
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0 or args.temperature < 0:
        raise ValueError("Invalid generation token budget or temperature")
    if getattr(args, "backend", "hf") == "openai":
        from .openai_teacher import OpenAITeacher
        if args.max_api_calls is None:
            raise ValueError("The API teacher requires an explicit --max-api-calls budget")
        if args.four_bit or args.revision or args.temperature != 0:
            raise ValueError("API teacher does not use --four-bit, --revision or --temperature; API seed control is unavailable")
        return OpenAITeacher(args.model, args.max_new_tokens, args.max_api_calls, args.reasoning_effort)
    if not args.revision:
        raise ValueError("A Hugging Face teacher requires an exact --revision")
    return HFGenerator(args.model, args.revision, args.max_input_tokens, args.max_new_tokens,
                       args.temperature, args.seed, args.four_bit, getattr(args, "adapter", None))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Biomedical data augmentation for downstream knowledge distillation")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Preserve official test IDs and create seeded train/dev/pilot splits")
    p.add_argument("--raw", required=True)
    p.add_argument("--test-labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dev-size", type=int, default=100)
    p.add_argument("--pilot-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("augment", help="Generate one three-stage augmentation route with resumable caching")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--route", required=True, choices=["cr", "bio"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--retry-errors", action="store_true", help="Explicitly retry failed cached model calls")
    model_options(p, api_teacher=True)
    p = commands.add_parser("import-chat", help="Validate and package attributed current-chat drafts without model or API calls")
    p.add_argument("--sources", required=True)
    p.add_argument("--drafts", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("review-pack", help="Export pending review records without automatically approving candidates")
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("build-sft", help="Build supervised examples from original data and reviewed candidates")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidates")
    p.add_argument("--reviews")
    p = commands.add_parser("train", help="Optional reader experiment: train a LoRA adapter on an allocated CUDA GPU")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("predict", help="Generate JSON predictions without exposing reference labels to the model")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    model_options(p, adapter=True)
    p = commands.add_parser("score", help="Compute task accuracy and macro F1 with strict ID checks")
    p.add_argument("--reference", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare
        result = prepare(args.raw, args.test_labels, args.output, args.dev_size, args.pilot_size, args.seed)
    elif args.command == "augment":
        from .augment import augment
        result = augment(args.input, args.output, args.route, generator(args), args.limit, args.seed, args.retry_errors)
    elif args.command == "review-pack":
        from .augment import review_pack
        result = review_pack(args.candidates, args.output)
    elif args.command == "import-chat":
        from .chat_import import import_chat
        result = import_chat(args.sources, args.drafts, args.output)
    elif args.command == "build-sft":
        from .data import build_sft
        result = build_sft(args.train, args.dev, args.output, args.candidates, args.reviews)
    elif args.command == "train":
        from .train import train
        result = train(args.train, args.dev, args.model, args.revision, args.output, args.steps, args.seed, args.max_length, args.four_bit)
    elif args.command == "predict":
        from .evaluate import predict
        result = predict(args.input, args.output, generator(args))
    else:
        from .evaluate import score
        result = score(args.reference, args.predictions, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
