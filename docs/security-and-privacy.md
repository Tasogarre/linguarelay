# Security and privacy

LinguaRelay assumes source media, transcripts, and provider-derived output may
be private.

- Process only media you are authorised to use.
- Keep job and model-cache roots outside the repository.
- Keep job directories owner-only and do not weaken their permissions.
- Supply provider credentials through environment variables.
- Never commit source media, output media, transcripts, logs, provider
  responses, cookies, browser profiles, or signed URLs.
- Review `git status --ignored` before every commit.

The local ingest command copies a regular, non-symlink media file into the job
workspace, marks it read-only, and records only a relative job path, byte
count, content hash, and local transport type. It does not persist the original
absolute path.

JSON written through the workspace API rejects sensitive field names and
URL-shaped values. Network provider errors intentionally omit response bodies
and credentials. These controls reduce accidental leakage; they do not replace
provider terms, copyright review, or local disk encryption.

See [SECURITY.md](../SECURITY.md) for vulnerability reporting.
