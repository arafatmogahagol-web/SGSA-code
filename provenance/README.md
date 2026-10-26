# Source provenance

- `source_manifest.json` records the origin and original SHA256 of imported files. It describes the source snapshots, before the packaging fixes documented in this release. Destinations use repository-relative paths. StemGNN is fetched separately and is described in `stemgnn_source.json`.
- `packaged_hashes.json` records current release-file SHA256 values after CRLF-to-LF normalization, so source verification works across Windows and Linux. The manifest excludes itself, Git internals, data, checkpoints, generated outputs and externally fetched StemGNN source.
- `release_checks.json` records the completed source, inference, checkpoint and cached-data checks, including their limits. It is not a new full training run.
- `development_history.json` preserves the original exploratory metadata; final execution scripts do not run a test-label oracle search.

Run `python tools/verify_sources.py` from any working directory to check the packaged hashes and Python syntax without importing models or starting training.
