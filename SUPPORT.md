# Support

## How to get help

| Channel | Use it for |
|---|---|
| [GitHub Issues](https://github.com/Siddhesh-ai-del/saafscan/issues) | Bugs, broken behaviour, feature requests |
| [GitHub Discussions](https://github.com/Siddhesh-ai-del/saafscan/discussions) | Questions about approach, design and usage |
| [Security advisories](https://github.com/Siddhesh-ai-del/saafscan/security/advisories/new) | Vulnerabilities only — never general support |

Before opening an issue, search existing issues and check the repository README.

## What to include

- The exact command or request you ran.
- Full error output or traceback.
- OS, Python version and install method.
- For detection problems: the photograph involved, with personal data redacted first.

## What not to expect

**There is no service-level agreement at this stage.** SaafScan is a pre-release student
prototype. Responses are best-effort and there is no guaranteed response time, no support
contract and no paid support tier.

## Common checks before reporting

1. Confirm the application starts: `bash start_demo.sh`, then open http://127.0.0.1:8000
2. Run the test suite: `.venv/bin/python -m pytest`
3. Run the linter: `ruff check src tests`
4. Confirm Python 3.12 is in use.

If the suite passes locally but a reading is wrong, include the photograph and the
diagnostics output:

```bash
.venv/bin/python -m saafscan.diagnostics photo.jpg --out debug/
```

## Commercial or research use

This project is not currently supported for commercial, clinical or regulatory use. See
the disclaimer in the README.
