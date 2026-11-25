#!/usr/bin/env python3
"""
query_claude.py -- Claude's answers to the project's friction prompts, by API, in a file the runners already read.

llm_prior.py asks OpenAI.  This asks Anthropic the same question: the prompt of every pair in the truth files, exactly as
run_pair_set.load_pairs builds it (llm_prior.build_prompt), with no system prompt, no tools and the same JSON shape
(median, low, high, reasoning; llm_prior.SCHEMA), validated by llm_prior.parse_response.  The answers go into the stand-in
file format, {"model", "entries": {sha256(prompt): {"prompt", "response", ...}}}, so run_pair_set.py, compare_llms.py and
the patchy-table study take the file as --llm NAME=FILE with no change.

    python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude.json            # preview: what would be asked; no request
    python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude.json --query    # paid: one request per pair not in the file
    python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude_s2.json --sample 2 --query   # a second sample, its own file
    python3 evaluation/query_claude.py --out data/llm_cache_handbook_haiku.json --model claude-haiku-5-5 --query

Rules, the same as the OpenAI path where the two APIs allow it:
  - Nothing is requested without --query.  Credentials come from the environment (ANTHROPIC_API_KEY); none is read from a file.
  - One file holds one model, one effort setting and one sample: one answer per prompt.  Another sample, model or effort
    is another file, and each file is its own method in the runners.
  - The file is saved after every answer, so a run that stops is continued by the same command.
  - A pair the model declined, or answered with numbers that fail parse_response, is recorded under "unanswered" and is
    not asked again unless --retry-unanswered is given: asking until an answer passes would be choosing among answers.
    The runners stop on a pair that has no answer, so leaving it out or asking again is a decision someone makes.
  - No fallback model.  An answer from another model would change which model the result is about.

What differs from the OpenAI settings, because the API does:
  - No temperature.  The current Claude models reject sampling parameters; run-to-run spread is seen by repeating (--sample).
  - Thinking is on (it cannot be switched off on claude-opus-5-5) and is not sent as a parameter; its depth is set by
    output_config.effort, which is passed explicitly and recorded (the default here is that model's own default, medium).
"""
import argparse, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from llm_prior import SCHEMA, parse_response
from run_pair_set import TRUTH, load_pairs, standin_key

MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 16000          # a ceiling, not a target: thinking and the answer together use a few hundred

class Unanswered(Exception):
    """The model returned no usable estimate: it declined, or its numbers fail llm_prior.parse_response."""
    def __init__(self, reason, record):
        super().__init__(reason)
        self.reason, self.record = reason, record

def request_settings(model=MODEL, effort="medium"):
    """Every request field that can change the answer, apart from the prompt.  Recorded in the file and compared on reuse."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    return {"model": model, "max_tokens": MAX_TOKENS,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}

def new_file(settings, sample):
    return {"model": settings["model"],
            "method": "Anthropic Messages API, one request per pair: the project prompt as the only user message, no system prompt, "
                      "no tools, no sampling parameters, thinking at the model's default with the effort below, JSON by "
                      "structured outputs, no fallback model.  One answer per prompt in this file (evaluation/query_claude.py).",
            "provider": "anthropic", "settings": settings, "sample": int(sample), "entries": {}, "unanswered": {}}

def load(path, settings, sample):
    """The file's contents, or a new header.  Refuses a file that is not this script's, or was made with other settings."""
    path = Path(path)
    if not path.exists():
        return new_file(settings, sample)
    data = json.loads(path.read_text())
    if data.get("provider") != "anthropic" or "settings" not in data:
        raise ValueError(f"{path} was not written by query_claude.py; use a new file")
    if data["settings"] != settings or data.get("sample") != int(sample):
        raise ValueError(f"{path} holds {data['settings']['model']} answers at effort {data['settings']['output_config']['effort']}, "
                         f"sample {data.get('sample')}; another model, effort or sample goes in another file")
    return data

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temp = f.name
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)

def ask(client, prompt, settings):
    """One request.  Returns the entry to store (the validated JSON text and what the response says about itself), or raises
    Unanswered with the record to keep.  A response cut off by max_tokens is a setup error and raises RuntimeError."""
    response = client.messages.create(messages=[{"role": "user", "content": prompt}], **settings)
    usage = getattr(response, "usage", None)
    record = {"prompt": prompt, "response_model": response.model, "stop_reason": response.stop_reason,
              "request_id": getattr(response, "_request_id", None), "created_utc": datetime.now(timezone.utc).isoformat(),
              "usage": {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens")}}
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise Unanswered("refusal", dict(record, reason="refusal", category=getattr(details, "category", None)))
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"the response stopped with {response.stop_reason!r}, not a complete answer; nothing was recorded")
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parse_response(raw)
    except ValueError as exc:                     # json.JSONDecodeError is a ValueError too
        raise Unanswered("malformed", dict(record, reason=f"malformed: {exc}", response=raw))
    return dict(record, response=raw)

def make_client():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("the anthropic package is not installed: pip install -r requirements.txt")
    return anthropic.Anthropic()                  # credentials from the environment; the SDK's default retries and timeout

def fill(path, pairs, model=MODEL, effort="medium", sample=1, query=False, retry_unanswered=False, client=None, out=print):
    """Ask for every pair that has no answer in the file.  Without query, only report what would be asked.
    Returns dict(cached, asked, answered, unanswered, pending)."""
    settings = request_settings(model, effort)
    data = load(path, settings, sample)
    keys = {p: standin_key(d["prompt"]) for p, d in pairs.items()}
    skipped = [] if retry_unanswered else [p for p in pairs if keys[p] in data["unanswered"] and keys[p] not in data["entries"]]
    missing = [p for p in pairs if keys[p] not in data["entries"] and p not in skipped]
    count = dict(cached=len(pairs) - len(missing) - len(skipped), asked=0, answered=0, unanswered=len(skipped), pending=len(missing))
    out(f"{path}: {count['cached']} of {len(pairs)} pairs answered, {len(skipped)} recorded as unanswered, {len(missing)} to ask "
        f"({model}, effort {effort}, sample {sample})")
    if skipped:
        out(f"  unanswered and not asked again without --retry-unanswered: {' '.join(skipped)}")
    if not query:
        if missing:
            out(f"  would ask: {' '.join(missing)}")
            out("  the first prompt, exactly as sent:\n" + "\n".join("    " + line for line in pairs[missing[0]]["prompt"].splitlines()))
        out("  nothing requested; add --query to make the paid requests")
        return count
    client = client if client is not None else make_client()
    for p in missing:
        count["asked"] += 1
        try:
            entry = ask(client, pairs[p]["prompt"], settings)
        except Unanswered as exc:
            data["unanswered"][keys[p]] = dict(exc.record, pair_when_asked=p)
            count["unanswered"] += 1
            out(f"  {p}: no estimate ({exc.record['reason']})")
        else:
            data["entries"][keys[p]] = dict(entry, pair_when_asked=p)
            data["unanswered"].pop(keys[p], None)
            count["answered"] += 1
            est = parse_response(entry["response"])
            out(f"  {p}: median {est['median']}, 90% interval {est['low']}-{est['high']}")
        count["pending"] -= 1
        save(path, data)
    return count

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the answers file; one per model, effort and sample")
    ap.add_argument("--model", default=MODEL, help=f"exact model ID (default {MODEL}); IDs are complete as written")
    ap.add_argument("--effort", choices=EFFORTS, default="medium", help="output_config.effort, recorded in the file")
    ap.add_argument("--sample", type=int, default=1, help="which repeat this file is; only a label, each sample is its own file")
    ap.add_argument("--truth", nargs="+", type=Path, default=list(TRUTH), help="truth files (measured-friction format)")
    ap.add_argument("--pairs", nargs="+", default=None, help="only these pair ids")
    ap.add_argument("--query", action="store_true", help="paid: make the requests")
    ap.add_argument("--retry-unanswered", action="store_true", help="ask again for pairs recorded as declined or malformed")
    a = ap.parse_args(argv)
    pairs = load_pairs(a.truth, a.pairs)
    try:
        count = fill(a.out, pairs, a.model, a.effort, a.sample, a.query, a.retry_unanswered)
    except ValueError as exc:
        raise SystemExit(str(exc))
    except TypeError as exc:                      # how the SDK reports that it found no credentials, before sending anything
        raise SystemExit(f"the request could not be sent: {exc}\nIf this is about authentication: set ANTHROPIC_API_KEY in this "
                         "terminal session and run the same command again.")
    except Exception as exc:
        # The SDK's typed errors, most specific first; answers received before the error are already saved.
        try:
            import anthropic
        except ImportError:
            raise exc
        if isinstance(exc, anthropic.AuthenticationError):
            raise SystemExit("no valid credentials: set ANTHROPIC_API_KEY in this terminal session")
        if isinstance(exc, anthropic.PermissionDeniedError):
            raise SystemExit(f"this key may not use {a.model}: {exc.message}")
        if isinstance(exc, anthropic.NotFoundError):
            raise SystemExit(f"model {a.model!r} was not found; check the ID")
        if isinstance(exc, anthropic.RateLimitError):
            raise SystemExit("rate limited; run the same command again in a minute, it continues where it stopped")
        if isinstance(exc, anthropic.APIStatusError):
            raise SystemExit(f"API error {exc.status_code}: {exc.message}")
        if isinstance(exc, anthropic.APIConnectionError):
            raise SystemExit("could not reach the API; run the same command again, it continues where it stopped")
        raise
    if a.query:
        print(f"asked {count['asked']}: {count['answered']} answered, {count['unanswered']} unanswered in all; saved to {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
