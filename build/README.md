# Building darkroom subjects

`Build subjects` (`.github/workflows/build.yml`) builds each subject listed in a batch file in `build/batches/` from its config in `build/targets/`, runs the same checks as the darkroom's own export (target fields, manifest schema, `check_text.py --strict`), and adds the results to a release named like `stock-batch-9`:

- `<id>.tar` holds the manifest, the layers and the subject's `target.json`, ready to upload;
- `<id>-check.tar` holds the check pictures and logs.

Nothing here uploads anywhere; going live is a separate run in the darkroom's own repository. The scripts in `build/scripts/` and `build/schema/` are copies of the darkroom's export code and are kept in step with it.
