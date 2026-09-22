# Optional post-answer citation audit

The normal assistant already reviews drafts using its Ollama model. This extra
button checks eligible generated claims against their attached document quotes
using the separately installed MiniCheck-Flan-T5-Large model. It does not change
answers, train the assistant, or establish that an answer is true. Missing model
assets produce an explicit unavailable result; ordinary chat still works.

This optional service retains the existing audited package identities. It uses
the CUDA PyTorch wheel **on CPU**, with accelerator access disabled in its private
worker. Its environment is therefore substantially larger than the main CPU-only
E5 environment. The original tested configuration was Windows and Python 3.11;
the same package wheels may not be available on other platforms. Platform wheel
availability was not rechecked online while preparing this source-only package.

From the repository root, with Python 3.11 installed:

```powershell
py -3.11 -m venv .audit-venv
.audit-venv\Scripts\python.exe -m pip install --upgrade pip
.audit-venv\Scripts\python.exe -m pip install -r requirements-audit.txt
python scripts/download_minicheck.py
```

The last command downloads approximately 3.14 GB of pinned official model files
plus the pinned upstream inference code, README and license. It resumes partial
downloads and verifies each file's size and SHA-256. It does not execute upstream
code during setup. Only three hash-verified upstream methods are loaded later by
the inference worker. No model weights are included in this repository.

The default paths are `.audit-venv/Scripts/python.exe` on Windows (or
`.audit-venv/bin/python` on POSIX) and `models/minicheck/manifest.json`. Override
them before starting the app when using an existing installation:

```powershell
$env:SISU_READER_AUDIT_PYTHON = "D:\my-audit-env\Scripts\python.exe"
$env:SISU_READER_AUDIT_MANIFEST = "D:\my-minicheck-cache\manifest.json"
python run_assistant.py ui
```

For offline transfers of a cache produced by this package:

```powershell
python scripts/download_minicheck.py --source "D:\verified-cache\minicheck"
python scripts/download_minicheck.py --verify-only
```

The source directory must contain `model/` and
`upstream--b58b9fa69acbd1015ec970fa65dd752413a053d2/`. The relative model path in
the installed manifest allows moving the complete cache together.

Start chat normally, ask a document question, and use the optional citation-check
button on a completed answer. Answer or diagnostic traces must be enabled, and
the current principal must retain source and trace access. The service checks
at most six eligible claim units, does not truncate inputs longer than 2,048
tokens into a score, and leaves unsupported representations visibly unchecked.
Its scoring-work budget is 20 seconds with a separate startup allowance of up to
60 seconds. A timed-out worker is stopped. The support score is the original
MiniCheck two-label softmax probability (threshold 0.5), not a calibrated truth
probability. Sources and permissions are checked again before release.

The `assistant_runtime/minicheck_*_cpu_run.py` files are internal compatibility
helpers retained from the source snapshot; their historical experiment entry
points are not application commands. Use the setup helper and browser button.

MiniCheck code is Apache-2.0 and its pinned model card declares MIT. See
`assistant_runtime/third_party/` for the retained code license, model card and
attribution. E5, Ollama and the generation model have their own licenses.
