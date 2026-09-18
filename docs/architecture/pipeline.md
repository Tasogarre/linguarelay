# Pipeline architecture

LinguaRelay turns an authorised local video into Brazilian Portuguese
transcript data, English translation data, subtitle files, and two rendered
videos. Private working data remains under a configured job root outside the
Git checkout.

```text
local media
  -> ingest + immutable receipt
  -> normalised audio
  -> ASR transcript
  -> validated English translation
  -> SRT + WebVTT
  -> soft-subtitled + burned-in video
  -> validation report + manifest
```

Each stage records output hashes in `state.json`. Resume mode accepts an output
only when its hash, stage context, and dependency ordering still match.
Persisted JSON passes through the redaction boundary in
`src/linguarelay/security.py`; credentials and URL-shaped metadata are
rejected.

The default configuration uses one ASR provider for a straightforward first
run. An explicit evaluation mode can compare multiple providers against a
manually reviewed synthetic or operator-supplied reference before full
transcription. Provider responses are parsed in memory and are not written to
the repository.

The pipeline is intentionally narrow in version 0.1: Brazilian Portuguese
(`pt-BR`) source speech to English (`en`) subtitles.
