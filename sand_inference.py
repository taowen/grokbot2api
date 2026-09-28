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
from collections.abc import Iterator
from pathlib import Path

DEFAULT_BACKEND_URL = "https://api2.cursor.sh"
RENEWAL_PATH = "/sand-box/inference-credential"
INFERENCE_PATH = "/aiserver.v1.InferenceService/Stream"
CREDENTIAL_ENV = "SAND_INFERENCE_RENEWAL_CREDENTIAL"

DEFAULT_CLIENT_TYPE = "sand"
DEFAULT_CLIENT_VERSION = "0.30.0"
DEFAULT_NAMESPACE = "prod"

# src/shared/agents/agent-model.ts
DEFAULT_MODEL_ID = "grok-4.7"
DEFAULT_MAX_MODE = False
DEFAULT_MODEL_PARAMS = (("effort", "high"), ("fast", "true"))

SECRETS_ENV_FILE = Path(
    os.environ.get("GROKBOT_SECRETS_FILE") or (Path.home() / ".omp/agent/secrets/grokbot.env")
)


def read_secrets_file() -> dict[str, str]:
    """omp's grokbot secrets file (SAND_INFERENCE_RENEWAL_CREDENTIAL + GROKBOT_MACHINE_ID).

    Keeps one source of truth: process env wins, this file is the fallback so the bridge can be
    started by a supervisor unit without duplicating secrets into unit config.
    """
    out: dict[str, str] = {}
    try:
        for line in SECRETS_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    except OSError:
        pass
    return out

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


def read_exactly(fp, count: int) -> bytes:
    """Read exactly ``count`` bytes from ``fp``, or fewer at end of stream."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = fp.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_envelopes_from_fp(fp) -> Iterator[tuple[int, bytes]]:
    """Yield (flags, payload) as each Connect envelope arrives on ``fp``.

    Same framing as :func:`iter_envelopes_from_bytes` — one flag byte, a 4-byte
    big-endian length, then the payload — but this one never waits for end of
    stream, so the bridge can forward a delta the moment upstream emits it. A
    truncated trailing envelope is dropped, exactly as the buffered reader drops
    it.
    """
    while True:
        header = read_exactly(fp, 5)
        if len(header) < 5:
            return
        length = int.from_bytes(header[1:5], "big")
        payload = read_exactly(fp, length)
        if len(payload) < length:
            return
        yield header[0], payload


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
    override = (os.environ.get("SAND_MACHINE_ID") or os.environ.get("GROKBOT_MACHINE_ID") or "").strip()
    if not override:
        override = (read_secrets_file().get("GROKBOT_MACHINE_ID") or "").strip()
    if override:
        return override

    candidates = [Path("/etc/machine-id")]
    appdata = os.environ.get("APPDATA")
    if appdata:
        # Reuse Cursor's own persisted device identity on Windows.  Generating
        # a UUID per request makes the upstream see every call as a new
        # computer and quickly triggers its device-count abuse protection.
        candidates.append(Path(appdata) / "Cursor" / "machineid")
    for path in candidates:
        try:
            machine_id = path.read_text().strip()
        except (OSError, UnicodeError):
            continue
        if machine_id:
            return machine_id

    # Last-resort deterministic identity for platforms without either file.
    # This is intentionally stable across requests and process restarts.
    node = uuid.getnode()
    return uuid.uuid5(uuid.NAMESPACE_OID, f"{node:012x}").hex


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


def write_cache(
    path: Path,
    access_token: str,
    expires_at_ms: int,
    conversation_id: str | None = None,
    session_token: str | None = None,
) -> None:
    payload: dict = {"accessToken": access_token, "expiresAtMs": expires_at_ms}
    prev = read_cache(path) or {}
    cid = conversation_id or prev.get("conversationId")
    if isinstance(cid, str) and cid:
        payload["conversationId"] = cid
    session = session_token or prev.get("sessionToken")
    if isinstance(session, str) and session:
        payload["sessionToken"] = session
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        # ``fchmod`` is unavailable on Windows.  The mode passed to os.open
        # already restricts the file on POSIX; repeat it there to correct an
        # existing cache file whose permissions may be broader.
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as cache_file:
            fd = -1
            cache_file.write(json.dumps(payload))
    finally:
        if fd >= 0:
            os.close(fd)


def load_renewal_credential(args) -> str:
    env = (os.environ.get(CREDENTIAL_ENV) or os.environ.get("GROKBOT_RENEWAL_CREDENTIAL") or "").strip()
    if not env:
        secrets = read_secrets_file()
        env = (secrets.get(CREDENTIAL_ENV) or secrets.get("GROKBOT_RENEWAL_CREDENTIAL") or "").strip()
    if env:
        return env
    raise SystemExit(f"set {CREDENTIAL_ENV} (or write {SECRETS_ENV_FILE}) before starting the client")


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
    # The sand exchange returns TWO tokens: accessToken (type: session) and grokBotToken
    # (type: grok_bot). Only grokBotToken authenticates aiserver.v1.InferenceService/Stream —
    # measured on 2026-09-15: accessToken yields ERROR_NOT_LOGGED_IN, grokBotToken streams.
    # STRICT: no fallback to the session token. A silent fallback would turn a missing/renamed
    # field into "every request fails with ERROR_NOT_LOGGED_IN"; fail loudly and name the field.
    token = None
    if isinstance(parsed, dict):
        for field in ("grokBotToken", "grok_bot_token"):
            candidate = parsed.get(field)
            if isinstance(candidate, str) and candidate:
                token = candidate
                break
    if not isinstance(token, str) or not token:
        keys = sorted(parsed.keys()) if isinstance(parsed, dict) else []
        raise SystemExit(
            "renewal returned no grokBotToken (response keys: "
            f"{keys}); refusing to fall back to the session token"
        )
    exp = parsed.get("expiresAtMs")
    if not isinstance(exp, (int, float)):
        exp = jwt_exp_ms(token) or (now_ms() + DEFAULT_TTL_MS)
    return {"accessToken": token, "expiresAtMs": int(exp), "renewed": True, "sessionToken": (parsed.get("accessToken") if isinstance(parsed, dict) else None)}


# Sand usage/allowance status. Lives on the Dashboard service, is an empty request message, and —
# unlike inference — authenticates with the SESSION token (accessToken), not grokBotToken:
#   measured 2026-09-15: accessToken -> 200, grokBotToken -> 401 ERROR_NOT_LOGGED_IN.
SAND_USAGE_PATH = "/aiserver.v1.DashboardService/GetSandUsageStatus"


def fetch_sand_usage(
    credential: str,
    backend_url: str,
    meta: dict[str, str],
    machine_id: str,
    timeout_ms: int = 30000,
) -> dict:
    """Mint a session token and read the sand weekly allowance status.

    Returns the upstream fields (usagePercent is USED, not remaining) plus
    usageRemainingPercent, or {"error": ...} — never raises for callers that want a soft surface.
    """
    try:
        minted = renew(credential, backend_url, meta)
    except SystemExit as e:
        return {"error": str(e)}
    session = minted.get("sessionToken")
    if not isinstance(session, str) or not session:
        return {"error": "renewal returned no session token for usage lookup"}
    url = backend_url.rstrip("/") + SAND_USAGE_PATH
    headers = {
        "authorization": f"Bearer {session}",
        "content-type": "application/json",
        "connect-protocol-version": "1",
        "connect-timeout-ms": str(timeout_ms),
        "x-cursor-checksum": cursor_checksum(machine_id),
        **meta,
    }
    req = urllib.request.Request(url, data=b"{}", method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"usage lookup HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"usage lookup {type(e).__name__}: {e}"}
    if not isinstance(data, dict):
        return {"error": "usage lookup returned a non-object payload"}
    used = data.get("usagePercent")
    out = {
        "usagePercent": used,
        "usageRemainingPercent": round(100 - float(used), 3) if isinstance(used, (int, float)) else None,
        "currentPeriodStart": data.get("currentPeriodStart"),
        "nextResetTimestampUtc": data.get("nextResetTimestampUtc"),
        "hasAvailableUsage": data.get("hasAvailableUsage"),
        "hasNonZeroIncludedLimit": data.get("hasNonZeroIncludedLimit"),
        "availableBankedResetCount": data.get("availableBankedResetCount"),
        "usesPooledEnterpriseAllowance": data.get("usesPooledEnterpriseAllowance"),
        "grokPlanLabel": data.get("grokPlanLabel"),
        "cursorPlanName": data.get("cursorPlanName"),
    }
    ondemand = data.get("onDemandSettings")
    if isinstance(ondemand, dict):
        out["onDemandDashboardUrl"] = ondemand.get("dashboardUrl")
    return out


def get_access_token(args, credential: str, meta: dict[str, str], force: bool = False) -> dict:
    cached = None if force else read_cache(args.cache)
    if cached and cache_valid(cached) and isinstance(cached.get("sessionToken"), str) and cached["sessionToken"]:
        return {
            "accessToken": cached["accessToken"],
            "expiresAtMs": int(cached["expiresAtMs"]),
            "renewed": False,
            "sessionToken": cached["sessionToken"],
        }
    got = renew(credential, args.backend_url, meta)
    write_cache(
        args.cache,
        got["accessToken"],
        got["expiresAtMs"],
        args.conversation_id,
        session_token=got.get("sessionToken") if isinstance(got.get("sessionToken"), str) else None,
    )
    return got

GROKBOT_SERVICE = "/aiserver.v1.GrokBotService"


def grokbot_json_headers(args, session_token: str, machine_id: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {session_token}",
        "content-type": "application/json",
        "connect-protocol-version": "1",
        "x-cursor-checksum": cursor_checksum(machine_id),
        **client_meta(args),
    }


def grokbot_rpc(args, session_token: str, method: str, body: dict, timeout_s: float = 25) -> dict:
    url = args.backend_url.rstrip("/") + f"{GROKBOT_SERVICE}/{method}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=grokbot_json_headers(args, session_token, load_machine_id()),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"{method} HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
        ) from e
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} returned a non-object")
    return parsed


def last_user_text(messages: list) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            text = "".join(parts).strip()
            if text:
                return text
    raise RuntimeError("no user text to send to Grok Bot")


def decode_transcript_body(body: object) -> dict | None:
    if not isinstance(body, str) or not body:
        return None
    import base64

    try:
        parsed = json.loads(base64.b64decode(body))
    except Exception:
        try:
            parsed = json.loads(body)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def transcript_text(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    message = parsed.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(parsed.get("content"), str):
        return parsed["content"]
    return ""


def resolve_grokbot_agent(args, session_token: str) -> dict:
    override = (os.environ.get("GROKBOT_AGENT_ID") or "").strip()
    payload = grokbot_rpc(args, session_token, "ListGrokBotAgents", {})
    agents = payload.get("agents") if isinstance(payload.get("agents"), list) else []
    if override:
        for agent in agents:
            if isinstance(agent, dict) and str(agent.get("id") or "") == override:
                return agent
        raise RuntimeError(f"GROKBOT_AGENT_ID={override} is not in ListGrokBotAgents")
    if not agents or not isinstance(agents[0], dict):
        raise RuntimeError("ListGrokBotAgents returned no agents")
    return agents[0]


def send_grokbot_user_message(
    args,
    session_token: str,
    messages: list,
    timeout_s: float = 90,
    on_event=None,
) -> dict:
    """Dispatch one user turn through GrokBotService and wait for the assistant row.

    Measured 2026-09-28: InferenceService/Stream returns ERROR_NOT_HIGH_ENOUGH_PERMISSIONS
    even from inside the sand VM. Live Grok Bot chat uses SendGrokBotUserMessage with the
    session token, then ListGrokBotTranscriptEntries on the agent's legacyAgentId.
    """
    prompt = last_user_text(messages)
    agent = resolve_grokbot_agent(args, session_token)
    numeric_id = str(agent.get("id") or "")
    legacy_id = str(agent.get("legacyAgentId") or numeric_id)
    if not numeric_id:
        raise RuntimeError("Grok Bot agent has no id")
    listed = grokbot_rpc(
        args, session_token, "ListGrokBotTranscriptEntries", {"agentId": legacy_id, "limit": 5}
    )
    before = 0
    for entry in listed.get("entries") or []:
        try:
            before = max(before, int(entry.get("seq") or 0))
        except (TypeError, ValueError):
            pass
    message_id = str(uuid.uuid4())
    send = grokbot_rpc(
        args,
        session_token,
        "SendGrokBotUserMessage",
        {
            "agentId": numeric_id,
            "messageId": message_id,
            "text": prompt,
            "sentAtMs": now_ms(),
        },
    )
    if send.get("dispatched") is not True:
        raise RuntimeError(f"SendGrokBotUserMessage did not dispatch: {send}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        page = grokbot_rpc(
            args,
            session_token,
            "ListGrokBotTranscriptEntries",
            {"agentId": legacy_id, "limit": 30},
        )
        rows = sorted(page.get("entries") or [], key=lambda entry: int(entry.get("seq") or 0))
        saw_user = False
        for entry in rows:
            try:
                seq = int(entry.get("seq") or 0)
            except (TypeError, ValueError):
                continue
            if seq <= before:
                continue
            parsed = decode_transcript_body(entry.get("body"))
            kind = entry.get("entryKind") or entry.get("entry_kind")
            text = transcript_text(parsed)
            if kind == "message" and isinstance(parsed, dict) and parsed.get("role") == "user":
                if prompt[:80] in text:
                    saw_user = True
                continue
            if saw_user and kind == "send-message" and text.strip():
                if on_event is not None:
                    on_event({"type": "text", "text": text})
                model = getattr(args, "model", DEFAULT_MODEL_ID)
                return {
                    "ok": True,
                    "httpStatus": 200,
                    "text": text,
                    "tool_calls": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "extended_usage": {},
                    "model": model,
                    "modelId": model,
                    "error": None,
                    "agentId": numeric_id,
                    "legacyAgentId": legacy_id,
                    "messageId": message_id,
                }
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for Grok Bot reply after {timeout_s}s")



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
