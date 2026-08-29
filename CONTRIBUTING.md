# Contributing

Contributions are welcome.

## Development setup

```bash
git clone https://github.com/taowen/grokbot2api.git
cd grokbot2api
python3 -m unittest discover -s tests -v
python3 -m py_compile grokbot2api.py responses_api.py api_common.py sand_inference.py
```

The runtime must remain dependency-free unless a dependency provides a clear interoperability or security benefit.

## Pull requests

- Keep all code, documentation, commit messages, and test names in English.
- Add offline tests for protocol changes.
- Do not include generated Cursor bundles or other private runtime artifacts.
- Do not include credentials, access tokens, request transcripts, or model output containing private data.
- Document reconstructed fields and observed behavior in `docs/protocol.md`.
- Clearly distinguish observed behavior from assumptions.

Live integration tests must be opt-in and must never run in public CI.
