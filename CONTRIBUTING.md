# Contributing

Thanks for helping improve LinguaRelay.

1. Open an issue before a large change so scope can be agreed.
2. Create a focused branch and include tests for changed behaviour.
3. Keep media, credentials, provider responses, logs, and job data outside the
   repository.
4. Run the checks below and open a pull request describing the user-visible
   change.

```bash
export UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/linguarelay/venv"
uv sync --locked --all-extras --dev
uv run pytest -q
uv build --no-sources
```

By contributing, you agree that your contribution is licensed under the MIT
License.
