# Source provenance

- `source_manifest.json` records the origin and original SHA256 of imported files. It
  describes the source snapshots, before the packaging fixes documented in this release.
  Destinations use repository-relative paths. StemGNN is fetched separately and is
  described in `stemgnn_source.json`.
- `packaged_hashes.json` records the current release-file SHA256 values after
  CRLF-to-LF normalization, so source verification works across Windows and Linux. The
  manifest excludes itself, Git internals, data, checkpoints, generated outputs and the
  externally fetched StemGNN source. It is regenerated with
  `python tools/update_manifest.py` after an intentional edit.
- `release_checks.json` is the dated record of the source, inference, checkpoint and
  cached-data checks performed for the original release, including their limits. It is
  not a new full training run.
- `smoke_checks.json` is the regenerable current-tree check: it parses every Python file,
  builds every architecture on CPU, compares parameter counts with the values printed in
  the paper, verifies that both configurations match the recorded splits, and calls
  `--help` on each entry point. Regenerate with `python tools/smoke_checks.py`.
  PDFormer and StemGNN are reported as skipped because their inputs (training-only prior
  file, separately fetched upstream source) are not redistributed. Because it carries a
  timestamp and is meant to be re-run, this file is deliberately excluded from
  `packaged_hashes.json`; the check is reproducible from `tools/smoke_checks.py` instead.
- `development_history.json` preserves the original exploratory metadata; final execution
  scripts do not run a test-label oracle search.

Run `python tools/verify_sources.py` from any working directory to check the packaged
hashes and Python syntax without importing models or starting training.
