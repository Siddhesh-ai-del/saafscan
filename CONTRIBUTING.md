# Contributing to SaafScan

Thank you for helping improve SaafScan. This document explains how to report problems,
propose changes and get work merged.

## Ground rules

- Be respectful and constructive. All participation is covered by the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- Discuss substantial changes before writing code. Open an issue describing the problem
  and the intended approach.
- Keep the project within its founding constraint: everything must work with a budget of
  £0 beyond consumables.

## Reporting a bug

Use the **Bug report** issue form. Before filing, search existing issues to avoid
duplicates.

A good report includes:

- What you did, what you expected and what happened instead.
- The exact command or request, and the full traceback.
- Environment details: OS, Python version and install method.
- A photograph or sample where a detection problem is involved.

**Redact personal data before attaching images.** Strip photographs can contain reflections,
surrounding rooms and documents. Crop or blur anything that is not the strip and backdrop.

## Requesting a feature

Use the **Feature request** form. Frame the request as the problem it solves, not the
implementation you have in mind, and note which project principle it advances.

## Development setup

Requires Python 3.12.

```bash
uv venv --python 3.12 .venv
uv sync --python .venv/bin/python
bash start_demo.sh
```

Open http://127.0.0.1:8000 to confirm the application starts before making changes.

## Testing — test-driven development is mandatory

Changes are expected to follow TDD: write the failing test first, confirm it fails, then
make it pass.

```bash
.venv/bin/python -m pytest
```

Run the suite plain. `pytest` is configured with `-q`, so adding your own `-q` suppresses
the summary line.

- **Coverage floor: 80%.** New logic needs tests that exercise both the success path and
  the refusal path.
- **Do not weaken a test to make it pass.** If a test is wrong, say so explicitly in the
  pull request and explain why.
- A refusal mechanism must never be removed to raise accuracy. The project's position is
  that a well-calibrated refusal is a correct outcome.

## Linting

```bash
ruff check src tests
```

Must be clean before a pull request.

## Commit messages

Use the conventional format:

```
<type>: <description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

## Pull requests

Complete the pull request template. A reviewer will check that:

1. Tests were added or updated and the suite passes.
2. Coverage remains at or above 80%.
3. `ruff check src tests` is clean.
4. No secrets, keys, tokens or personal data are present.
5. Numbers quoted in documentation trace back to a source file.

## Project-specific constraints

- **Printed colour targets are never ingestible as chemistry.** Folder labels record which
  swatch a print depicts, never a test result.
- **Never crop or adjust pixels in a photographed validation frame** to make a classifier
  agree. That changes the answer, not the photograph.
- **Splits are decided by session, never by photograph.** All frames from one sitting go to
  one split.
- Presentation decks are built by script. Never hand-edit a `.pptx`.

## Security issues

Do not open a public issue for vulnerabilities. Follow [SECURITY.md](SECURITY.md).
