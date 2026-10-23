## Type of change

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `refactor` — behaviour-preserving restructure
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `chore` / `perf` / `ci`

## What changed and why

<!-- Describe the problem first, then the approach. Link the issue if there is one. -->

## Checklist

- [ ] Tests were added or updated, written **before** the implementation (TDD)
- [ ] Full suite passes: `.venv/bin/python -m pytest`
- [ ] Coverage remains at or above **80%**
- [ ] `ruff check src tests` is clean
- [ ] No secrets, API keys, tokens or credentials are present
- [ ] No personal data or unredacted photographs are committed
- [ ] Commit messages follow the conventional format
- [ ] Any number quoted in docs traces back to a source file

## Behaviour and refusal paths

- [ ] New behaviour has a test covering the **success** path
- [ ] New behaviour has a test covering the **refusal** path
- [ ] No refusal mechanism was removed in order to raise accuracy

## Visual changes

<!-- Delete this section if not applicable. -->

- [ ] Deck or chart changes were produced by running the build script, not by hand-editing output
- [ ] The verification gate passes: `python deck/verify_final_deck.py`
