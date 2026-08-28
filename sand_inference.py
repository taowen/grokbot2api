#!/usr/bin/env python3
"""Renew an inference token and call Cursor's private inference stream.

Default model: grok-4.5 with effort=high and fast=true.
The conversation ID is generated once and retained in the token cache.

Usage:
  SAND_INFERENCE_RENEWAL_CREDENTIAL=... python3 sand_inference.py "ping"
  SAND_INFERENCE_RENEWAL_CREDENTIAL=... python3 sand_inference.py --renew-only
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import ssl
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BACKEND_URL = "https://api2.cursor.sh"
RENEWAL_PATH = "/sand-box/inference-credential"
INFERENCE_PATH = "/aiserver.v1.InferenceService/Stream"
CREDENTIAL_ENV = "SAND_INFERENCE_RENEWAL_CREDENTIAL"

DEFAULT_CLIENT_TYPE = "sand"
DEFAULT_CLIENT_VERSION = "0.30.0"
DEFAULT_NAMESPACE = "prod"

# src/shared/agents/agent-model.ts
DEFAULT_MODEL_ID = "grok-4.5"
DEFAULT_MAX_MODE = False
DEFAULT_MODEL_PARAMS = (("effort", "high"), ("fast", "true"))

REFRESH_LEEWAY_MS = 2 * 60 * 1000
DEFAULT_TTL_MS = 10 * 60 * 1000
DEFAULT_CACHE = Path(os.environ.get("SAND_TOKEN_CACHE") or "/tmp/grokbot2api-direct-token.json")

ROLE = {"unspecified": 0, "user": 1, "assistant": 2, "tool": 3, "system": 4}


def _varint(n: int) -> bytes:
    out = bytearray()
    n = n & ((1 << 64) - 1)
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def pb_str(field: int, s: str) -> bytes:
    b = s.encode("utf-8")
    return _key(field, 2) + _varint(len(b)) + b


def pb_msg(field: int, inner: bytes) -> bytes:
    return _key(field, 2) + _varint(len(inner)) + inner


def pb_var(field: int, n: int) -> bytes:
    return _key(field, 0) + _varint(n)


def pb_bool(field: int, v: bool) -> bytes:
    return pb_var(field, 1 if v else 0)


def pb_decode(buf: bytes, i: int = 0, end: int | None = None) -> tuple[dict[int, list], int]:
    if end is None:
        end = len(buf)
    fields: dict[int, list] = {}
    while i < end:
        n, i = _read_varint(buf, i)
        field, wire = n >> 3, n & 7
        if wire == 0:
            val, i = _read_varint(buf, i)
        elif wire == 1:
            val, i = buf[i : i + 8], i + 8
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            val, i = buf[i : i + ln], i + ln
        elif wire == 5:
            val, i = buf[i : i + 4], i + 4
        else:
            break
        fields.setdefault(field, []).append(val)
    return fields, i


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    n = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if b < 0x80:
            return n, i
        shift += 7
    return n, i


def encode_core_message(role: str, text: str) -> bytes:
    return pb_var(1, ROLE[role]) + pb_str(2, text)


def encode_param(pid: str, value: str) -> bytes:
    return pb_str(1, pid) + pb_str(2, value)


def encode_requested_model(model_id: str, max_mode: bool, params: list[tuple[str, str]]) -> bytes:
    inner = pb_str(1, model_id)
    if max_mode:
        inner += pb_bool(2, True)
    for pid, value in params:
        inner += pb_msg(3, encode_param(pid, value))
    return inner


def encode_stream_request(
    messages: list[tuple[str, str]],
    model_id: str,
    max_mode: bool,
    params: list[tuple[str, str]],
    invocation_id: str,
    conversation_id: str,
) -> bytes:
    body = b""
    for role, text in messages:
        body += pb_msg(1, encode_core_message(role, text))
    if model_id:
        body += pb_msg(7, encode_requested_model(model_id, max_mode, params))
    if invocation_id:
        body += pb_str(6, invocation_id)
    if conversation_id:
        body += pb_str(8, conversation_id)
        body += pb_str(12, conversation_id)
    return body


def decode_text_parts(payload: bytes) -> dict:
    fields, _ = pb_decode(payload)
    out: dict = {"text": "", "thinking": "", "error": None, "model": None}
    for raw in fields.get(1, []):
        f, _ = pb_decode(raw)
        if 1 in f:
            out["text"] += f[1][0].decode("utf-8", "replace")
    for raw in fields.get(9, []):
        f, _ = pb_decode(raw)
        if 1 in f:
            out["thinking"] += f[1][0].decode("utf-8", "replace")
    for raw in fields.get(4, []):
        f, _ = pb_decode(raw)
        if 2 in f:
            out["model"] = f[2][0].decode("utf-8", "replace")
    for raw in fields.get(8, []):
        f, _ = pb_decode(raw)
        out["error"] = f[1][0].decode("utf-8", "replace") if 1 in f else repr(f)
    return out


def connect_envelope(payload: bytes, flags: int = 0) -> bytes:
    return bytes([flags]) + len(payload).to_bytes(4, "big") + payload


def iter_envelopes_from_bytes(raw: bytes) -> list[tuple[int, bytes]]:
    chunks: list[tuple[int, bytes]] = []
    i = 0
    while i + 5 <= len(raw):
        flags = raw[i]
        ln = int.from_bytes(raw[i + 1 : i + 5], "big")
        i += 5
        if i + ln > len(raw):
            break
        chunks.append((flags, raw[i : i + ln]))
        i += ln
    return chunks


def enhanced_obfuscate(data: bytearray) -> bytes:
    last = 165
    for i in range(len(data)):
        data[i] = ((data[i] ^ last) + (i % 256)) & 255
        last = data[i]
    return bytes(data)


def cursor_checksum(machine_id: str) -> str:
    import base64

    unix_kilo = int(time.time() * 1000 // 1_000_000)
    raw = bytearray(
        [
            (unix_kilo >> 40) & 255,
            (unix_kilo >> 32) & 255,
            (unix_kilo >> 24) & 255,
            (unix_kilo >> 16) & 255,
            (unix_kilo >> 8) & 255,
            unix_kilo & 255,
        ]
    )
    checksum = base64.urlsafe_b64encode(enhanced_obfuscate(raw)).decode("ascii").rstrip("=")
    return f"{checksum}{machine_id}"


def load_machine_id() -> str:
    p = Path("/etc/machine-id")
    if p.is_file():
        return p.read_text().strip()
    return uuid.uuid4().hex


def client_meta(args) -> dict[str, str]:
    return {
        "x-cursor-client-type": args.client_type,
        "x-cursor-client-version": args.client_version,
        "x-sand-box-namespace": args.namespace,
    }


def jwt_exp_ms(token: str) -> int | None:
    try:
        import base64

        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = data.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp * 1000)
    except Exception:
        return None
    return None


def now_ms() -> int:
    return int(time.time() * 1000)


def cache_valid(cache: dict, leeway_ms: int = REFRESH_LEEWAY_MS) -> bool:
    token = cache.get("accessToken")
    exp = cache.get("expiresAtMs")
    if not isinstance(token, str) or not token:
        return False
    if not isinstance(exp, int):
        exp = jwt_exp_ms(token)
    if not isinstance(exp, int):
        return False
    return now_ms() < exp - leeway_ms


def read_cache(path: Path) -> dict | None:
    try:
        if path.is_symlink():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_cache(path: Path, access_token: str, expires_at_ms: int, conversation_id: str | None = None) -> None:
    payload: dict = {"accessToken": access_token, "expiresAtMs": expires_at_ms}
    prev = read_cache(path) or {}
    cid = conversation_id or prev.get("conversationId")
    if isinstance(cid, str) and cid:
        payload["conversationId"] = cid
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as cache_file:
            fd = -1
            cache_file.write(json.dumps(payload))
    finally:
        if fd >= 0:
            os.close(fd)


def load_renewal_credential(args) -> str:
    env = (os.environ.get(CREDENTIAL_ENV) or "").strip()
    if env:
        return env
    raise SystemExit(f"set {CREDENTIAL_ENV} before starting the client")


def default_conversation_id() -> str:
    return str(uuid.uuid4())


def resolve_conversation_id(args) -> str:
    if args.conversation_id:
        return args.conversation_id.strip()
    cached = read_cache(args.cache) or {}
    cid = cached.get("conversationId")
    if isinstance(cid, str) and cid:
        return cid
    return default_conversation_id()


def renew(credential: str, backend_url: str, meta: dict[str, str]) -> dict:
    url = urllib.parse.urljoin(backend_url.rstrip("/") + "/", RENEWAL_PATH.lstrip("/"))
    body = json.dumps({"credential": credential}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"content-type": "application/json", **meta},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"renewal failed HTTP {e.code}: {e.read().decode('utf-8','replace')[:500]}") from e
    token = parsed.get("accessToken") if isinstance(parsed, dict) else None
    if not isinstance(token, str) or not token:
        raise SystemExit("renewal returned no accessToken")
    exp = parsed.get("expiresAtMs")
    if not isinstance(exp, (int, float)):
        exp = jwt_exp_ms(token) or (now_ms() + DEFAULT_TTL_MS)
    return {"accessToken": token, "expiresAtMs": int(exp), "renewed": True}


def get_access_token(args, credential: str, meta: dict[str, str], force: bool = False) -> dict:
    cached = None if force else read_cache(args.cache)
    if cached and cache_valid(cached):
        return {
            "accessToken": cached["accessToken"],
            "expiresAtMs": int(cached["expiresAtMs"]),
            "renewed": False,
        }
    got = renew(credential, args.backend_url, meta)
    write_cache(args.cache, got["accessToken"], got["expiresAtMs"], args.conversation_id)
    return got


def inference_headers(args, access_token: str, machine_id: str, request_id: str) -> dict[str, str]:
    h = {
        "authorization": f"Bearer {access_token}",
        "content-type": "application/connect+proto",
        "connect-protocol-version": "1",
        "connect-timeout-ms": str(args.timeout_ms),
        "x-cursor-checksum": cursor_checksum(machine_id),
        "x-ghost-mode": "true",
        "x-request-id": request_id,
        **client_meta(args),
    }
    if args.team_id:
        h["x-cursor-team-id"] = str(args.team_id)
    return h


def stream_llm(args, access_token: str, prompt: str) -> dict:
    machine_id = load_machine_id()
    request_id = str(uuid.uuid4())
    invocation_id = str(uuid.uuid4())
    proto = encode_stream_request(
        [("user", prompt)],
        args.model,
        args.max_mode,
        list(DEFAULT_MODEL_PARAMS),
        invocation_id,
        args.conversation_id,
    )
    body = connect_envelope(proto, 0)

    parsed = urllib.parse.urlparse(args.backend_url)
    host = parsed.hostname or "api2.cursor.sh"
    port = parsed.port or (443 if parsed.scheme != "http" else 80)
    if parsed.scheme != "http":
        conn = http.client.HTTPSConnection(host, port, timeout=args.timeout_ms / 1000, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=args.timeout_ms / 1000)
    headers = inference_headers(args, access_token, machine_id, request_id)
    conn.request("POST", INFERENCE_PATH, body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    status = resp.status
    content_type = resp.getheader("content-type") or ""
    conn.close()

    if status != 200:
        return {
            "ok": False,
            "httpStatus": status,
            "contentType": content_type,
            "error": raw[:800].decode("utf-8", "replace"),
            "requestId": request_id,
            "modelId": args.model,
            "conversationId": args.conversation_id,
        }

    envelopes = iter_envelopes_from_bytes(raw)
    texts: list[str] = []
    thinking: list[str] = []
    model = None
    errors: list[str] = []
    trailer = None
    for flags, payload in envelopes:
        if flags & 2:
            try:
                trailer = json.loads(payload.decode("utf-8") or "{}")
            except Exception:
                trailer = {"raw": payload[:400].decode("utf-8", "replace")}
            continue
        if flags & 1:
            payload = gzip.decompress(payload)
        part = decode_text_parts(payload)
        if part["text"]:
            texts.append(part["text"])
        if part["thinking"]:
            thinking.append(part["thinking"])
        if part["model"]:
            model = part["model"]
        if part["error"]:
            errors.append(part["error"])

    err = None
    if trailer and isinstance(trailer, dict) and trailer.get("error"):
        err = trailer["error"]
    if errors and err is None:
        err = errors[0]

    return {
        "ok": err is None,
        "httpStatus": status,
        "requestId": request_id,
        "model": model or args.model,
        "modelId": args.model,
        "conversationId": args.conversation_id,
        "text": "".join(texts),
        "thinking": "".join(thinking),
        "error": err,
        "envelopes": len(envelopes),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-renew sand inference token and call InferenceService.Stream")
    p.add_argument("prompt", nargs="?", help="user prompt; omit with --renew-only")
    p.add_argument("--backend-url", default=os.environ.get("SAND_BACKEND_URL") or os.environ.get("CURSOR_API_BASE_URL") or DEFAULT_BACKEND_URL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--force-renew", action="store_true")
    p.add_argument("--renew-only", action="store_true")
    p.add_argument("--model", default=os.environ.get("SAND_MODEL") or DEFAULT_MODEL_ID)
    p.add_argument("--max-mode", action="store_true", help="opt-in; paid plan only (ERROR_RATE_LIMITED_CHANGEABLE otherwise)")
    p.add_argument("--conversation-id", default=os.environ.get("SAND_CONVERSATION_ID") or "")
    p.add_argument("--client-type", default=DEFAULT_CLIENT_TYPE)
    p.add_argument("--client-version", default=DEFAULT_CLIENT_VERSION)
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--team-id", default=os.environ.get("SAND_TEAM_ID") or "")
    p.add_argument("--timeout-ms", type=int, default=120000)
    p.add_argument("--show-token", action="store_true", help="include the access token in --renew-only output")
    args = p.parse_args()
    args.max_mode = bool(args.max_mode) if getattr(args, "max_mode", False) else DEFAULT_MAX_MODE
    args.conversation_id = resolve_conversation_id(args)

    if not args.renew_only and not args.prompt:
        p.error("prompt is required unless --renew-only")

    meta = client_meta(args)
    credential = load_renewal_credential(args)
    tok = get_access_token(args, credential, meta, force=args.force_renew)

    if args.renew_only:
        out = {
            "ok": True,
            "renewed": tok["renewed"],
            "expiresAtMs": tok["expiresAtMs"],
            "expiresInSec": max(0, (tok["expiresAtMs"] - now_ms()) // 1000),
            "cache": str(args.cache),
            "modelId": args.model,
            "conversationId": args.conversation_id,
        }
        if args.show_token:
            out["accessToken"] = tok["accessToken"]
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    result = stream_llm(args, tok["accessToken"], args.prompt)
    if result.get("httpStatus") == 401:
        tok = get_access_token(args, credential, meta, force=True)
        result = stream_llm(args, tok["accessToken"], args.prompt)
        result["renewedAfter401"] = True

    result["renewed"] = tok["renewed"]
    result["tokenExpiresInSec"] = max(0, (tok["expiresAtMs"] - now_ms()) // 1000)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    if result.get("text"):
        sys.stderr.write("\n----- text -----\n")
        sys.stderr.write(result["text"])
        sys.stderr.write("\n")
    if not result.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()
