# Security Policy

## Reporting a vulnerability

Report security vulnerabilities privately using
[GitHub security advisories](https://github.com/Siddhesh-ai-del/saafscan/security/advisories/new)
on this repository. This opens a private report visible only to the maintainers.

Please do not open a public issue for a suspected vulnerability.

Include:

- A description of the issue and its potential impact.
- Steps to reproduce, or a proof of concept.
- The affected version or commit.
- Any suggested remediation, if you have one.

You will receive an acknowledgement as soon as practicable. Once an issue is confirmed, a
fix will be prepared and credit given in the advisory unless you prefer otherwise.

## Scope

In scope:

- The HTTP API and static web frontend.
- File upload validation, including image size and type handling.
- The evidence store and PDF/QR report export.
- Injection or traversal risks in any path derived from a photograph or request field.
- Leakage of evidence records or personal data beyond the local machine.

Out of scope:

- Issues requiring physical access to the device running the prototype.
- Denial of service against a locally bound `127.0.0.1` service.
- The accuracy of adulterant classification. Classification quality is a correctness and
  safety concern rather than a security vulnerability; report it as a bug.
- Vulnerabilities in upstream dependencies with an existing published advisory.

## Supported versions

This is a pre-release student prototype. Only the current commit on `main` is supported.
No versioned releases are published, and no backports are issued.

## Design guarantees

The following are intentional properties of the design. If you find a way to break one,
that is a security bug.

- **No telemetry.** The application makes no outbound network requests of its own. There is
  no analytics, no crash reporting and no cloud inference.
- **Local binding by default.** The server binds `127.0.0.1` and is not exposed to the
  network unless explicitly configured otherwise.
- **No personal data leaves the machine.** Evidence records and photographs stay in local
  storage on the device that captured them.
- **Append-only evidence.** The store exposes no update or delete API for evidence records,
  so a logged reading cannot be quietly rewritten.
- **Refusal over fabrication.** The pipeline returns a structured HTTP 422 rather than a
  guessed verdict when a calibration card is absent or the frame cannot be trusted.
- **Server-side validation.** Image size, type and structure are validated on the server;
  client-side checks are a convenience, never the control.
