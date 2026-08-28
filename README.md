# grokbot2api

`grokbot2api` is a local compatibility proxy that lets [Grok Build](https://github.com/xai-org/grok-build) use Cursor-hosted Grok models through an OpenAI-compatible Chat Completions endpoint.

It translates Grok Build requests into Cursor's undocumented `aiserver.v1.InferenceService.Stream` protobuf messages. Messages, native tool calls, tool results, and multi-turn state are preserved across the bridge.

> [!WARNING]
> This project uses an undocumented Cursor endpoint and private protobuf schema. It is not affiliated with, endorsed by, or supported by Cursor, Anysphere, xAI, or Grok Build. The integration can stop working without notice. Use only credentials issued to you and only in ways permitted by the applicable service terms.

## Features

- OpenAI-compatible `POST /v1/chat/completions`
- Native Cursor protobuf tool calls and tool results
- Multi-turn Grok Build agent loops
- Streaming SSE responses with heartbeats
- `GET /v1/models` and `GET /health`
- Automatic short-lived access-token renewal
- Loopback-only binding by default
- No third-party Python runtime dependencies

## Requirements

- Linux
- Python 3.10 or newer
- [Grok Build](https://github.com/xai-org/grok-build) installed
- Access to a supported Cursor-hosted model such as `grok-4.6`
- A valid `SAND_INFERENCE_RENEWAL_CREDENTIAL` issued to you

## Install

```bash
git clone https://github.com/taowen/grokbot2api.git
cd grokbot2api
chmod 700 grokbot2api.py sand_inference.py
```

The proxy uses only Python's standard library.

## Configure Grok Build

Add the following model entry to `~/.grok/config.toml`:

```toml
[model.cursor-grok-4-6]
name = "Cursor Grok 4.6 via grokbot2api"
model = "grok-4.6"
base_url = "http://127.0.0.1:8765/v1"
api_backend = "chat_completions"
api_key = "local-only"
context_window = 256000
```

To make it the default model, also add:

```toml
[models]
default = "cursor-grok-4-6"
```

Verify that Grok Build sees the model:

```bash
grok models
```

### Context window

Cursor documents Cursor Grok 4.6 with a 256K context window. This differs from the 500K window advertised for `grok-4.6` on xAI's direct API; `grokbot2api` uses Cursor's inference route, so the configuration uses `256000`. System instructions, tool schemas, reasoning, conversation history, and tool results all consume part of that window.

## Start the proxy

Do not put the renewal credential directly in shell history. Read and export it interactively:

```bash
read -rsp "Cursor renewal credential: " SAND_INFERENCE_RENEWAL_CREDENTIAL
echo
export SAND_INFERENCE_RENEWAL_CREDENTIAL
./grokbot2api.py
```

Expected output:

```text
grokbot2api listening on http://127.0.0.1:8765/v1
model: grok-4.6; upstream: /path/to/grokbot2api/sand_inference.py
```

Check the local endpoint:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/models
```

## Use Grok Build

In another terminal:

```bash
cd /path/to/your/project
grok -m cursor-grok-4-6
```

Headless example:

```bash
grok -m cursor-grok-4-6 -p "Inspect this project and explain how it works."
```

Multi-turn tool-call check:

```bash
grok -m cursor-grok-4-6 \
  --permission-mode bypassPermissions \
  --max-turns 8 \
  -p 'Run pwd, then run git status --short in a second tool call, then report both results.'
```

Use an appropriate permission mode for your environment. `bypassPermissions` is shown only to make the automated example non-interactive.

## Command-line options

```text
--listen ADDRESS          Listen address (default: 127.0.0.1)
--port PORT               Listen port (default: 8765)
--model MODEL             Default upstream model (default: grok-4.6)
--upstream-script PATH    Path to sand_inference.py
--backend-url URL         Override the Cursor backend URL
--cache PATH              Short-lived access-token cache
--max-mode                Enable max mode when the account supports it
--conversation-id ID      Override the upstream conversation ID
--timeout-ms MS           Upstream timeout (default: 120000)
--api-key-env NAME        Environment variable used to protect the local proxy
```

Run `./grokbot2api.py --help` for the complete list.

## Protecting the local endpoint

The server binds to `127.0.0.1` by default. To require a bearer token even on loopback:

```bash
read -rsp "Local proxy API key: " GROKBOT2API_KEY
echo
export GROKBOT2API_KEY
./grokbot2api.py --api-key-env GROKBOT2API_KEY
```

Set the same value as `api_key` in the Grok Build model configuration.

The proxy refuses to bind to a non-loopback address unless the selected API-key environment variable is non-empty. Exposing this service to a network is strongly discouraged.

## Architecture

```text
Grok Build
  OpenAI Chat Completions + SSE
        |
        v
grokbot2api
  message/tool schema bridge
        |
        v
Cursor api2 backend
  Connect protocol + private protobuf
        |
        v
Cursor-hosted Grok model
```

The outer API is OpenAI-compatible because that is Grok Build's custom-model interface. The upstream side uses Cursor's native inference message and tool-call structures rather than asking the model to emit a custom JSON protocol.

See [docs/protocol.md](docs/protocol.md) for the wire-format reference.

## Known limitations

- The Cursor inference API and protobuf schema are private and undocumented.
- `grok-4.6` currently rejects a present `InferenceAgentTool.parameters` protobuf field with provider status 422. The proxy omits that field and appends a compact argument signature to each native tool description. Tool calls and results still use native protobuf messages.
- Upstream responses are buffered by the helper before they are converted to SSE. Heartbeats keep Grok Build's connection alive, but token deltas are not forwarded in real time.
- Image content in Chat Completions messages is not currently forwarded.
- Token usage may be reported as zero when the private endpoint omits usage frames.
- Only Chat Completions is implemented; `/v1/responses` and `/v1/messages` are not.

## Troubleshooting

### `SAND_INFERENCE_RENEWAL_CREDENTIAL` is not set

Export a valid credential before starting the proxy. Never commit it, paste it into an issue, or include it in logs.

### Provider status 422

The upstream provider rejected a tool or model configuration. Confirm that you are running the latest proxy version, which omits the incompatible protobuf `parameters` field.

### `BrokenPipeError`

Older versions waited for the complete upstream response before opening the SSE stream. Current versions send headers immediately and emit heartbeat comments while waiting.

### Grok Build cannot find the model

Run `grok models` and validate `~/.grok/config.toml`. The table name, default model alias, and `-m` argument must match exactly.

## Security

Read [SECURITY.md](SECURITY.md) before deploying or modifying the proxy. In particular:

- Treat renewal credentials and cached access tokens as secrets.
- Keep the service on loopback.
- Rotate any credential pasted into a terminal transcript, chat, issue, or log.
- Do not publish request bodies; Grok Build may send source code and tool output.

## Development

Run the offline test suite:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile grokbot2api.py sand_inference.py
```

No live credential is required for tests.

## License

[MIT](LICENSE)
