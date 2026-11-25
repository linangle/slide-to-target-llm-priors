# Getting the language-model answers by API

Written 25 Nov 2026. How to replace the stand-in answers with real API calls (decision D16): what to fix before the first request, how to get and handle keys, the commands, and what to commit afterwards.

**Where things stand.**

| | GPT | Claude |
|---|---|---|
| Query code | `src/llm_prior.py` (`query_llm`), driven by `evaluation/run_pair_set.py --query-llm` | `evaluation/query_claude.py` (written 25 Nov; never yet run against the API) |
| Answers on disk | Y01–Y03 only (`data/llm_cache_published_pilot.json`) | 23 pairs, but a stand-in: one Claude Code subagent per pair, not an API call (`data/llm_standin_claude_haiku.json`) |
| Package in `.venv` | `openai` is pinned in `requirements.txt` but **not installed** | `anthropic` 2.14.0, pinned in `requirements.txt` and installed 25 Nov |
| Key in the shell | Not set | Not set |

So each half is a matter of a key and a few commands (sections 4 and 5). Neither has been run against its API for the 23 pairs.

**Who runs what.** The paid requests are run by a person, in their own terminal, with their own key. Claude Code can run everything that needs no key: the checks before, and the comparison and the Drake runs after. Do not paste a key into a chat, a file in the repository, or a commit.

---

## 1. Freeze these before the first request

The cache identifies an answer by the prompt text and the request settings together (`llm_prior.prompt_key`). Changing either makes a new entry and leaves the old one in the file, so decide first and query once.

1. **The pairs and their descriptions.** A prompt is built only from `description_object` and `description_surface` (`llm_prior.build_prompt`; `tests/test_prompt_hygiene.py` fails if any other column reaches it).
   - L01–L20: all 20 have descriptions (`data/ground_truth_handbook.csv`).
   - Y01–Y03: the fixed strings in `evaluation/demo_published_comparison.py`.
   - Rig pairs M01–M27, V01–V03: **none of the 30 has a description yet.** Write them into the measured file's description columns, and commit them, before anyone looks at measured friction.
2. **The prompt wording.** One wording exists (`llm_prior.PROMPT_TEMPLATE`). The proposal planned two. A second wording is a second template and a second set of cache files; decide whether it is in scope.
3. **The models, by exact string.** GPT: `gpt-5.4-mini-2026-03-17` is what the code and the saved pilot use. Claude: pick one and write it down; `claude-opus-5-5` is the current default model, and `claude-haiku-5-5` is the tier the stand-in used. Claude model IDs are complete as written; do not add a date suffix.
4. **The request settings.** GPT: the ones in `llm_prior.request_settings()` (no reasoning effort, 600 output tokens, strict JSON schema, nothing stored). Claude: see section 5.
5. **How many samples per pair.** D16 says several. The cache holds **one answer per prompt per file**, so several samples means several files, one per sample (section 4, step 5).

Record the five choices in the run's notes. The methods section needs them: model strings, dates of the requests, settings, number of samples, and the fact that neither model had tools or web access.

## 2. Keys

**OpenAI.** Sign in to the OpenAI API platform (platform.openai.com), add billing or credit, and create a secret key under API keys. The SDK reads it from `OPENAI_API_KEY`.

**Anthropic.** Sign in to the Claude Console (platform.claude.com), add credit, and create a key under API keys. The SDK reads it from `ANTHROPIC_API_KEY`.

(The console screens were not checked on 25 Nov; the variable names are what the SDKs read.)

**Handling.**
- Put a monthly spending limit on each key or project. This job costs a few dollars at most (section 6).
- Set the key in the terminal session that runs the query, and nowhere else:

```bash
export OPENAI_API_KEY="..."        # typed in your own terminal; not saved to a file in the repository
```

- `.gitignore` has no `.env` entry. If you keep keys in a `.env` file, add `.env` to `.gitignore` first.
- A key that reaches a commit, a screenshot or a chat is compromised: revoke it in the console and make a new one.
- One key per person is cleaner than a shared one; the cache files are what the team shares, and a cache hit needs no key.

## 3. Install

```bash
source .venv/bin/activate
pip install -r requirements.txt      # installs the pinned openai package, which is missing from .venv today
```

`requirements.txt` pins both SDKs (`openai`, `anthropic`).

## 4. GPT: the commands

1. **Check the pair list without spending anything.** Needs Drake, no key.

```bash
python3 evaluation/run_pair_set.py --check-only
```

2. **Look at one prompt exactly as it will be sent.** Without `--query` this only prints.

```bash
python3 src/llm_prior.py --object "A clean, dry block with a hardened-steel contacting face" --surface "A clean, dry flat plate of mild steel"
```

3. **Request the answers.** One paid request per pair that is missing from the file, then it stops; no simulation runs. It prints each pair's median and interval as it goes. A request that fails is not retried and nothing malformed is cached; run the same command again and it continues from where it stopped.

```bash
python3 evaluation/run_pair_set.py --query-llm data/llm_cache_handbook_gpt.json
```

   Another OpenAI model goes in its own file: add `--model NAME`. A file holds one model only, and the runner refuses to mix.

4. **Check what came back,** from the caches, with no simulation and no key:

```bash
python3 evaluation/compare_llms.py --llm gpt=data/llm_cache_handbook_gpt.json claude=data/llm_standin_claude_haiku.json
```

5. **More samples per pair.** Repeat step 3 into a new file per sample (`..._gpt_s2.json`, `..._gpt_s3.json`, …). Each file is then its own method (`--llm gpt1=... gpt2=...`), so `compare_llms.py` shows the spread between samples directly. Pooling the samples into one prior (the proposal averaged medians in log space) is a few lines that do not exist yet.

6. **Commit the cache files** with the date and model in the commit message. From then on nobody queries again: the evaluators never request on a cache miss, and they record each cache file's hash in the run's `settings.json`.

## 5. Claude: the commands

`evaluation/query_claude.py` asks Anthropic the same question the OpenAI path asks: the prompt of every pair exactly as the runner builds it, as the only user message, with no system prompt and no tools, and the same JSON shape, checked by `llm_prior.parse_response`. It writes the stand-in file format, so the runners take the file as `--llm NAME=FILE` unchanged. `llm_prior.py`, which the evaluations hash, is not touched.

1. **Preview.** No key, no request, nothing written; it prints which pairs would be asked and the first prompt as sent.

```bash
python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude.json
```

2. **Request the answers** (paid). The file is saved after every answer; if the run stops, the same command continues.

```bash
python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude.json --query
```

3. **Another sample, model or effort is another file.** The script refuses to mix them in one.

```bash
python3 evaluation/query_claude.py --out data/llm_cache_handbook_claude_s2.json --sample 2 --query
```

```bash
python3 evaluation/query_claude.py --out data/llm_cache_handbook_haiku.json --model claude-haiku-5-5 --query
```

4. **Probe sets and the direct-speed condition.** `evaluation/probes.py` builds five further prompt sets from the same descriptions: a second wording, hidden material, counterfactual changes, recite-the-table, and C1 (the model is asked for the paddle speed itself). They are drafts: run `python3 evaluation/probes.py` to read them, change what should change, and commit the file before sending any. Each kind goes in its own file, because the answer's shape is part of the settings.

```bash
python3 evaluation/query_claude.py --probe hidden --out data/llm_probe_hidden_claude.json --query
```

   The five sets are 230 prompts per model and sample (23 + 23 + 92 + 23 + 69). Only Claude can be asked this way today; `llm_prior.py` has no probe path for GPT.

What the script fixes, and why it differs from GPT's settings:

- **Model:** `claude-opus-5-5` unless `--model` says otherwise. The ID is recorded in the file, along with the model each response names.
- **No temperature.** The current Claude models reject `temperature`, `top_p` and `top_k`, so "temperature 0" is not available as a way to pin the answer. Repeated samples show the run-to-run spread, which is what the proposal wanted anyway.
- **Thinking stays on** (it cannot be switched off on `claude-opus-5-5`) and is not sent as a parameter. Its depth is `output_config.effort`, passed explicitly and recorded: `--effort`, default `medium`, that model's own default. GPT's requests use no reasoning effort, so the two are not like for like on this point; say so in the methods.
- **JSON by structured outputs** with the project's schema.
- **No fallback model.** An answer from another model would change which model the result is about.
- **Declined or malformed answers are recorded, not replaced.** They go under `unanswered` in the file with the reason, and are not asked again unless you pass `--retry-unanswered`; asking until an answer passes would be choosing among answers. The runners stop on a pair with no answer, so dropping the pair or asking again is a decision you make and report. (The OpenAI path differs: it stops on a malformed answer and caches nothing, so a rerun asks again.)
- **Recorded with each answer:** the responding model, the stop reason, the request id, the time, and the token counts.

Checked without a key: `tests/test_query_claude.py` (nine tests), one of which sends a request through the real SDK to a server on localhost and compares the body the SDK sent with the settings above. Not checked: a request to the real API. The first paid run is the first real one, so start with `--pairs L01` and look at the answer before running the rest.

## 6. Cost and time

A prompt is about 110 tokens and an answer a few hundred, plus thinking on Claude. At `claude-opus-5-5`'s list price ($4 and $20 per million input and output tokens) and at most about 1,000 output tokens a call, 23 pairs × 5 samples is about 115 calls and roughly $2; with the 30 rig pairs and a second wording, about 530 calls and roughly $11. `claude-haiku-5-5` is about forty times cheaper. These are estimates, not measurements. GPT's cost was not looked up; it is the same order or less for a mini model. Either set of requests takes minutes.

## 7. After the caches exist

```bash
python3 evaluation/compare_llms.py --llm gpt=data/llm_cache_handbook_gpt.json claude=data/llm_cache_handbook_claude.json --out results/side_base/pair_sets/llm_comparison_api.txt
python3 evaluation/run_pair_set.py --llm gpt=data/llm_cache_handbook_gpt.json claude=data/llm_cache_handbook_claude.json --out results/side_base/pair_sets/handbook_api
```

The first needs no simulation. The second is the full comparison in Drake at the runner's default arm setup. For the rerun's setup, the trial wrapper takes its own switches, then `--`, then the runner's options:

```bash
python3 evaluation/run_pair_set_trial.py --a-brake 6 --check-accel --posture limits --one-piece -- --llm gpt=data/llm_cache_handbook_gpt.json claude=data/llm_cache_handbook_claude.json --seeds 0 1 2 3 4 5 6 7 8 9 --out results/side_base/pair_sets/rerun_api
```

Keep the stand-in file and its results as they are, labelled as a stand-in; new answers go in new files and new result folders.

## 8. Checklist

- [ ] Pairs, descriptions, wording, model strings, settings and sample count written down
- [ ] Rig descriptions committed before measured friction is seen (if the rig pairs are in scope)
- [ ] Keys created, spending limits set, keys only in the terminal session
- [ ] `--check-only` passes; one prompt previewed
- [ ] GPT cache filled, one file per sample
- [ ] Claude: one pair first (`--pairs L01 --query`), then the cache filled, one file per sample
- [ ] `compare_llms.py` report saved
- [ ] Cache files committed with model and date; no further requests
