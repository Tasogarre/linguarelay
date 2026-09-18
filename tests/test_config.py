from pathlib import Path

import pytest

from linguarelay.config import ConfigError, load_config


def test_load_config_expands_external_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = load_config(Path("config/example.toml"), repo_root=Path.cwd())

    assert config.jobs_root == home / ".local/share/linguarelay/jobs"
    assert config.source_language == "pt-BR"
    assert config.subtitle_policy.max_chars_per_line == 42


def test_rejects_jobs_root_inside_repository(tmp_path: Path) -> None:
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(
        """
schema_version = 1
[paths]
jobs_root = "./jobs"
model_cache_root = "~/.cache/models"
[languages]
source = "pt-BR"
target = "en"
[subtitles]
max_chars_per_line = 42
max_lines = 2
max_chars_per_second = 20.0
min_cue_seconds = 1.0
max_cue_seconds = 7.0
minimum_gap_seconds = 0.08
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="outside the repository"):
        load_config(config_path, repo_root=tmp_path)


def test_rejects_unsupported_language_pair(tmp_path: Path) -> None:
    config_path = tmp_path / "unsupported.toml"
    baseline = Path("config/example.toml").read_text(encoding="utf-8")
    config_path.write_text(
        baseline.replace(
            'jobs_root = "~/.local/share/linguarelay/jobs"',
            f'jobs_root = "{tmp_path.parent / "external-jobs"}"',
        ).replace('source = "pt-BR"', 'source = "es"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="supports only pt-BR"):
        load_config(config_path, repo_root=tmp_path)
