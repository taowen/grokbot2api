# Security Policy

## Credentials

`SAND_INFERENCE_RENEWAL_CREDENTIAL` and cached access tokens are secrets.

- Never commit credentials or token caches.
- Never paste credentials into issues, pull requests, screenshots, logs, or chat transcripts.
- Prefer interactive hidden input over putting a credential directly in shell history.
- Rotate a credential immediately if it is exposed.
- Use only credentials issued to you and only for authorized access.

The public project intentionally does not scan other processes, Cursor runtime files, or `/proc/*/environ` for credentials.

## Network exposure

The proxy binds to `127.0.0.1` by default. Keep it on loopback.

If a non-loopback bind is explicitly selected, the proxy requires a bearer token supplied through the environment variable named by `--api-key-env`. This check is a minimum safeguard, not a recommendation to expose the service.

Requests may contain source code, system prompts, tool arguments, and tool output. Do not place an untrusted reverse proxy in front of the service.

## Reporting a vulnerability

Please open a GitHub security advisory for vulnerabilities that do not require including live credentials or private request data. Rotate affected credentials before reporting.

Do not open a public issue containing secrets, access tokens, source code, or private Cursor runtime artifacts.
