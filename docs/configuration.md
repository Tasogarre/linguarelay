# Configuration

`config/example.toml` is the supported baseline. Copy it when you need local
changes; do not put credentials in TOML.

## Paths

`paths.jobs_root` must resolve outside the Git checkout. LinguaRelay rejects a
repository-local root and creates job directories with owner-only permissions.

## Providers

`asr.mode = "single"` selects the sole name in `asr.candidates` for the
normal first-run path. The provider definition names the environment variable
containing its credential. Translation is configured independently under
`translation`.

For an advanced bake-off, set `asr.mode = "evaluate"`, configure at least two
unique candidates, set `require_manually_reviewed_reference = true`, ingest
the media, and add `asr/reference.json` under the job root. The reference uses
the same transcript shape as `tests/fixtures/reference.json`, plus
`manually_reviewed = true` and a non-empty `reviewed_by` value. A normal
`linguarelay run` then includes the preparation and evaluation stages
automatically.

## Media and subtitles

The media section controls FFmpeg/FFprobe commands, audio normalisation, and
chunk duration. An empty `subtitles.font_path` uses Pillow's portable default
font. Set an explicit TrueType font path when consistent typography across
machines matters.

Quality thresholds are deterministic release gates. A failed threshold blocks
the final PASS manifest; lower them only when you understand the resulting
quality trade-off.
