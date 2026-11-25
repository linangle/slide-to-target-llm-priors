"""query_claude.py: Claude's answers by API into the stand-in file format.  No network and no key: a fake client stands in
for the SDK, and the last test sends one request through the real SDK to a server on localhost (skipped without the package)."""
import json, shutil, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace as NS
import pytest
import llm_prior
import query_claude as Q
import run_pair_set as R
from paths import DATA

GOOD = json.dumps({"median": 0.3, "low": 0.2, "high": 0.45, "reasoning": "steel on steel, dry"})
BAD = json.dumps({"median": 0.3, "low": 0.5, "high": 0.4, "reasoning": "interval out of order"})

def response(text=GOOD, stop_reason="end_turn", model="claude-opus-5-5"):
    blocks = [NS(type="thinking", thinking=""), NS(type="text", text=text)]
    return NS(content=blocks, stop_reason=stop_reason, model=model, _request_id="req_test",
              usage=NS(input_tokens=110, output_tokens=240), stop_details=NS(category="cyber") if stop_reason == "refusal" else None)

class Fake:
    """messages.create(**kwargs) -> the response chosen by `answer(prompt)`; keeps every call."""
    def __init__(self, answer=lambda prompt: response()):
        self.calls, self.answer = [], answer
        self.messages = NS(create=self.create)
    def create(self, **kw):
        self.calls.append(kw)
        return self.answer(kw["messages"][0]["content"])

@pytest.fixture
def pairs():
    return R.load_pairs(R.TRUTH, ["L01", "L02", "Y01"])

def quiet(*a, **k):
    pass

def test_without_query_nothing_is_requested_or_written(tmp_path, pairs):
    path, lines = tmp_path/"c.json", []
    count = Q.fill(path, pairs, client=Fake(lambda p: pytest.fail("a request was made")), out=lines.append)
    assert count["asked"] == 0 and count["pending"] == 3 and not path.exists()
    text = "\n".join(lines)
    assert "would ask: L01 L02 Y01" in text and "nothing requested" in text
    assert pairs["L01"]["prompt"].splitlines()[1] in text          # the prompt is shown as it would be sent

def test_request_is_the_project_prompt_and_nothing_that_api_rejects(tmp_path, pairs):
    fake = Fake()
    Q.fill(tmp_path/"c.json", pairs, effort="low", query=True, client=fake, out=quiet)
    assert len(fake.calls) == 3
    for kw, p in zip(fake.calls, pairs.values()):
        assert kw["messages"] == [{"role": "user", "content": p["prompt"]}]          # the runner's prompt, byte for byte, and only it
        assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == Q.MAX_TOKENS
        assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": llm_prior.SCHEMA}}
        assert set(kw) == {"messages", "model", "max_tokens", "output_config"}       # no temperature, thinking, system, tools, fallbacks

def test_answers_are_read_by_the_runner_and_a_rerun_asks_nothing(tmp_path, pairs):
    path, fake = tmp_path/"c.json", Fake()
    count = Q.fill(path, pairs, sample=2, query=True, client=fake, out=quiet)
    assert count == dict(cached=0, asked=3, answered=3, unanswered=0, pending=0)
    for p in pairs.values():
        assert R.llm_answer(p["prompt"], path) == json.loads(GOOD)
    assert R.cache_model(path) == "claude-opus-5-5"
    data = json.loads(path.read_text())
    entry = data["entries"][R.standin_key(pairs["L01"]["prompt"])]
    assert data["sample"] == 2 and data["settings"] == Q.request_settings() and data["provider"] == "anthropic"
    assert entry["pair_when_asked"] == "L01" and entry["response_model"] == "claude-opus-5-5" and entry["stop_reason"] == "end_turn"
    assert entry["request_id"] == "req_test" and entry["usage"] == {"input_tokens": 110, "output_tokens": 240} and entry["created_utc"]
    again = Fake(lambda p: pytest.fail("asked again"))
    assert Q.fill(path, pairs, sample=2, query=True, client=again, out=quiet)["asked"] == 0
    more = R.load_pairs(R.TRUTH, ["L01", "L02", "L03", "Y01"])                       # one new pair: one request
    assert Q.fill(path, more, sample=2, query=True, client=fake, out=quiet)["asked"] == 1 and len(fake.calls) == 4

def test_a_stopped_run_keeps_what_it_had(tmp_path, pairs):
    def answer(prompt):
        if prompt == pairs["L02"]["prompt"]:
            raise ConnectionError("network down")
        return response()
    path = tmp_path/"c.json"
    with pytest.raises(ConnectionError):
        Q.fill(path, pairs, query=True, client=Fake(answer), out=quiet)
    assert list(json.loads(path.read_text())["entries"]) == [R.standin_key(pairs["L01"]["prompt"])]
    assert Q.fill(path, pairs, query=True, client=Fake(), out=quiet)["asked"] == 2    # the same command continues

def test_declined_and_malformed_answers_are_recorded_and_not_asked_again(tmp_path, pairs):
    def answer(prompt):
        if prompt == pairs["L01"]["prompt"]:
            return response(stop_reason="refusal", text="")
        return response(BAD) if prompt == pairs["L02"]["prompt"] else response()
    path, fake = tmp_path/"c.json", Fake(answer)
    count = Q.fill(path, pairs, query=True, client=fake, out=quiet)
    assert count["answered"] == 1 and count["unanswered"] == 2
    data = json.loads(path.read_text())
    k1, k2 = (R.standin_key(pairs[p]["prompt"]) for p in ("L01", "L02"))
    assert set(data["entries"]) == {R.standin_key(pairs["Y01"]["prompt"])}
    assert data["unanswered"][k1]["reason"] == "refusal" and data["unanswered"][k1]["category"] == "cyber"
    assert data["unanswered"][k2]["reason"].startswith("malformed") and data["unanswered"][k2]["response"] == BAD
    assert R.llm_answer(pairs["L01"]["prompt"], path) is None                        # so a run that needs L01 stops, as for any missing pair
    lines = []
    assert Q.fill(path, pairs, query=True, client=Fake(lambda p: pytest.fail("asked again")), out=lines.append)["asked"] == 0
    assert "L01 L02" in "\n".join(lines)
    retry = Q.fill(path, pairs, query=True, retry_unanswered=True, client=Fake(), out=quiet)      # an explicit decision to ask again
    assert retry["asked"] == 2 and retry["answered"] == 2 and json.loads(path.read_text())["unanswered"] == {}

def test_a_truncated_answer_stops_the_run_and_records_nothing(tmp_path, pairs):
    path = tmp_path/"c.json"
    with pytest.raises(RuntimeError, match="max_tokens"):
        Q.fill(path, pairs, query=True, client=Fake(lambda p: response(GOOD[:20], stop_reason="max_tokens")), out=quiet)
    assert not path.exists()

def test_one_file_holds_one_model_effort_and_sample(tmp_path, pairs):
    path = tmp_path/"c.json"
    Q.fill(path, pairs, query=True, client=Fake(), out=quiet)
    for other in (dict(model="claude-haiku-5-5"), dict(effort="high"), dict(sample=2)):
        with pytest.raises(ValueError, match="another file"):
            Q.fill(path, pairs, query=True, client=Fake(), out=quiet, **other)
    standin = tmp_path/"standin.json"
    shutil.copy(DATA/"llm_standin_claude_haiku.json", standin)                       # never add API answers to the stand-in file
    with pytest.raises(ValueError, match="not written by query_claude"):
        Q.fill(standin, pairs, query=True, client=Fake(), out=quiet)
    with pytest.raises(ValueError, match="effort"):
        Q.request_settings(effort="none")

def test_command_line_previews_by_default_and_names_the_key_when_credentials_are_missing(tmp_path, monkeypatch, capsys):
    path = tmp_path/"c.json"
    monkeypatch.setattr(Q, "make_client", lambda: pytest.fail("a client was made without --query"))
    assert Q.main(["--out", str(path), "--pairs", "L01"]) == 0 and not path.exists()
    assert "nothing requested" in capsys.readouterr().out
    def no_credentials(prompt):
        raise TypeError("Could not resolve authentication method")
    monkeypatch.setattr(Q, "make_client", lambda: Fake(no_credentials))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        Q.main(["--out", str(path), "--pairs", "L01", "--query"])
    assert not path.exists()
    monkeypatch.setattr(Q, "make_client", lambda: Fake())
    assert Q.main(["--out", str(path), "--pairs", "L01", "--query"]) == 0 and R.llm_answer(R.load_pairs(R.TRUTH, ["L01"])["L01"]["prompt"], path)
    with pytest.raises(SystemExit, match="another file"):
        Q.main(["--out", str(path), "--pairs", "L01", "--effort", "high", "--query"])

def test_one_request_through_the_real_sdk_to_a_local_server(pairs):
    """The SDK accepts the settings as keyword arguments, sends them as the Messages body, and ask() reads its response
    objects.  A server on localhost answers; no key and no network are used."""
    anthropic = pytest.importorskip("anthropic")
    seen = {}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            body = json.dumps({"id": "msg_local", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                               "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": GOOD}],
                               "stop_reason": "end_turn", "stop_sequence": None,
                               "usage": {"input_tokens": 110, "output_tokens": 240}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json"); self.send_header("request-id", "req_local")
            self.send_header("content-length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="local-test-value", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0)
        prompt = pairs["L01"]["prompt"]
        entry = Q.ask(client, prompt, Q.request_settings())
    finally:
        server.shutdown()
    assert seen["path"].startswith("/v1/messages")
    assert seen["body"] == {"messages": [{"role": "user", "content": prompt}], **Q.request_settings()}
    assert json.loads(entry["response"]) == json.loads(GOOD) and entry["response_model"] == "claude-opus-5-5"
    assert entry["request_id"] == "req_local" and entry["usage"] == {"input_tokens": 110, "output_tokens": 240}
