# Packaging audit

This folder was prepared from the working V10 assistant for a user-managed Git upload. No remote repository was changed and no commit was pushed. Publishing is a separate step run by the user.

## Scope

- The application source, browser assets, pinned model-download manifests, attribution files, synthetic tests and setup instructions belong in the upload.
- The user's document collection, index, access database, conversations, saved answers, research datasets, evaluation outputs, virtual environments and model weights are excluded.
- `SOURCE_SNAPSHOT.json` records where the original source snapshot came from. Its relative research-tree references are provenance, not paths needed to run the exported application.

## Source and dependency checks

The copied Python files were parsed and their relative imports checked against this folder. No missing relative-import target was found. Core third-party imports are supplied by NumPy, Beautiful Soup, python-docx, pypdf, PyTorch and Transformers; the setup requirements also pin the tokenizer and safe-tensor dependencies used by the embedding loader. PyMuPDF is used only by the optional PDF OCR worker. The optional MiniCheck worker has its own dependency environment, including psutil. Tests use NumPy and pytest.

The document engine retains inherited infrastructure modules used by authorization, resource routing and diagnostics. Their presence does not enable continuous learning in V10: the active engine rejects the old strategy-learning controller. Historical evaluation and release utilities inside the package are internal source, not the installation or upload workflow for this folder.

## Privacy and content inspection

The source inspection checked for personal home-directory paths, the original local username, likely credential literals, private keys, databases, answer traces, document datasets and large model binaries. No such private content was found. Credential-detection expressions in `sisu_reader/release.py` are regular-expression definitions, not credentials. Public model URLs and third-party license/attribution records are intentional.

This is an inspection of the prepared source folder, not a guarantee about files added later. Inspect `git status --short` and `git diff --cached --stat` before committing; keep generated workspaces and model caches out of Git.

## Verification boundaries

The standalone checks should resolve `sisu_reader` and `assistant_runtime` from this folder with no inherited research `PYTHONPATH`. Synthetic regression tests exercise source binding, permission changes, retrieval-cache identity and request-budget behavior without a real language-model call. Runtime smoke checks separately exercise setup, indexing and the browser facade. Passing these checks does not establish answer correctness for every document or hardware platform.

The exported defaults and asset paths are adapted for this directory layout. Retrieval, evidence-binding and answer-review mathematics are preserved. Model weights must be downloaded or explicitly supplied by the user before full semantic retrieval is available; a missing encoder produces the declared keyword-only fallback.
