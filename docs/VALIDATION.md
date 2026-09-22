# Standalone export validation

The export was checked locally on Windows with Python 3.11. The working Python
environment and already downloaded E5/Ollama models were reused. Preparation
did not perform a fresh dependency installation or re-download the models.

## Completed checks

- Imports resolved inside this export with the inherited `PYTHONPATH` removed.
- **103 core regression tests passed**, covering retrieval, source binding,
  permission/source changes, answer revisions and request budgets.
- **16 model-setup tests passed**, covering verified local copying, download
  resumption, corrupted/oversized artifacts, invalid ranges, unsafe paths and
  optional model configuration. Download responses were simulated.
- The documented command examples parsed against the exported CLI.
- The runner successfully indexed the synthetic sample handbook into an
  isolated validation workspace and passed `doctor`.
- A real local-model answer correctly located the weekly notes at
  `onboarding/weekly-notes` and attached a source citation.
- The exported web facade started, reported the V10 pipeline as ready, and
  served its HTML, JavaScript and CSS. This was an HTTP startup check, not a new
  full browser interaction benchmark.
- A separate source/import/privacy audit found no missing relative imports,
  private documents, stored conversations, credentials or model binaries in
  the prepared source selection.

The live assistant in the original research workspace was not modified by
packaging. The packaged changes adapt filesystem paths and startup defaults;
they do not change retrieval scores, prompts, evidence selection or model
training. Exact source and export hashes are recorded in `SOURCE_SNAPSHOT.json`
and `PACKAGING.json` at the repository root.

## Limits of these checks

The tests do not establish answer quality on all documents. Model downloads,
wheel availability on every operating system, optional OCR accuracy and a fresh
installation of the optional MiniCheck environment were not retested here.
The source package contains setup procedures and pinned asset hashes for those
installations, not the downloaded assets themselves.
