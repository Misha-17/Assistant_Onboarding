# Publish this copy yourself

No GitHub connector or access granted to the coding assistant is required. The
commands below use Git on **your computer**, with your own GitHub credentials.
Nothing has been pushed during preparation.

This approach works with an empty repository or one that already has commits.
It makes a fresh clone, keeps its Git history, and replaces the tracked file tree
with this current assistant. It does not force-push or rewrite existing commits.
The `git rm` line affects only the fresh clone; the removal is part of the new
commit and previous files remain available in older commits.

Run this block in PowerShell from the directory containing
`Assistant_Onboarding_upload`. The destination must not already exist. If it
does, choose a different unused destination name rather than deleting it.

```powershell
$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath ".\Assistant_Onboarding_upload").Path
$checkout = Join-Path (Get-Location).Path "Assistant_Onboarding"
if (-not (Test-Path -LiteralPath "$source\README.md")) { throw "Upload folder is missing." }
if (Test-Path -LiteralPath $checkout) { throw "Destination already exists; choose a new checkout path." }

git clone https://github.com/Misha-17/Assistant_Onboarding.git $checkout
if ($LASTEXITCODE -ne 0) { throw "Git clone failed; no files were changed." }
Set-Location -LiteralPath $checkout
$actualRoot = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($actualRoot) -ne [IO.Path]::GetFullPath($checkout)) { throw "Unexpected repository directory." }

git rm -r --ignore-unmatch -- .
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the fresh checkout." }
robocopy $source $checkout /E /XD .git .venv .audit-venv __pycache__ .pytest_cache workspace workspaces models research_results /XF *.pyc *.pyo *.log *.sqlite3 *.sqlite3-* .env .env.*
if ($LASTEXITCODE -ge 8) { throw "File copy failed." }

git add --all
if ($LASTEXITCODE -ne 0) { throw "Staging failed." }
git diff --cached --stat
git status --short
git commit -m "Add standalone SISU V10 assistant and complete documentation"
if ($LASTEXITCODE -ne 0) { throw "Commit failed; read Git's message before continuing." }
git push -u origin HEAD
if ($LASTEXITCODE -ne 0) { throw "Push failed; read Git's message. Do not force-push." }
```

`git push -u origin HEAD` pushes the branch checked out by the clone, so you do
not need to guess whether the repository uses `main` or another branch name.
An empty repository may print a warning when cloned; that is normal.

If Git asks for your commit identity, set it **in this checkout** and retry the
commit and push:

```powershell
git config user.name "Your name"
git config user.email "Your GitHub-associated or GitHub noreply email"
```

If Git asks you to sign in, authenticate through your own Git credential manager.
Do not paste credentials into the README, source files, or commands shared with
others. If branch protection rejects a direct push, create a branch and push it:

```powershell
git switch -c upload-sisu-v10
git push -u origin upload-sisu-v10
```

Then open a pull request on GitHub. For a non-fast-forward rejection, fetch and
reconcile the new remote changes; do not use `--force`.
