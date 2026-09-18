# LinguaRelay

LinguaRelay is a privacy-conscious Python pipeline that turns an authorised
Brazilian Portuguese video into an English translation, SRT and WebVTT
subtitles, and soft-subtitled and burned-in videos.

Version 0.1 deliberately supports one language direction: `pt-BR` to `en`.
It works with local media and keeps source files, transcripts, logs, and output
under an external owner-only job directory.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe on `PATH`
- An OpenAI API key for the default ASR and translation configuration

Only process media you are authorised to use. Provider usage may incur charges.

## Setup

Keep uv's environment outside the checkout:

```bash
export UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/linguarelay/venv"
uv sync --locked --all-extras --dev
```

Export credentials in your shell; never write them into TOML or commit them:

```bash
export OPENAI_API_KEY="..."
```

The baseline [configuration](config/example.toml) stores jobs under
`~/.local/share/linguarelay/jobs`.

## First run

Ingest one local video you are authorised to process:

```bash
uv run linguarelay ingest \
  --config config/example.toml \
  --job-id my-first-job \
  --input /path/to/video.mp4
```

The command copies the source into the private job workspace and records a
content hash without persisting the original absolute path.

Run the pipeline:

```bash
uv run linguarelay run \
  --config config/example.toml \
  --job-id my-first-job \
  --resume
```

Successful jobs contain:

- `transcript/pt-BR.raw.json`
- `translation/en.json`
- `subtitles/en.srt` and `subtitles/en.vtt`
- `output/soft-subtitled.mp4` (or `.mkv` when MP4 muxing is unavailable)
  and `output/burned-in.mp4`
- `manifest.json` with `quality_status: "PASS"`

Inspect redacted progress with:

```bash
uv run linguarelay status \
  --config config/example.toml \
  --job-id my-first-job
```

The default single-provider route is the shortest usable path. The
[configuration guide](docs/configuration.md) explains the optional multi-ASR
evaluation mode.

## Development

```bash
export UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/linguarelay/venv"
uv sync --locked --all-extras --dev
uv run pytest -q
uv build --no-sources
```

Tests use synthetic fixtures and do not require provider credentials.

## Documentation

- [Pipeline architecture](docs/architecture/pipeline.md)
- [Configuration](docs/configuration.md)
- [Security and privacy](docs/security-and-privacy.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Licence

LinguaRelay is available under the [MIT License](LICENSE).
