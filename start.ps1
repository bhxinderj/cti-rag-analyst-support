param(
    [ValidateSet("hybrid", "bm25", "vector")]
    [string]$Mode = "hybrid",
    [string]$Question
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if ($Question) {
    & $VenvPython "main.py" "query" $Question "--mode" $Mode
} else {
    & $VenvPython "main.py" "interactive" "--mode" $Mode
}

if ($LASTEXITCODE -ne 0) {
    throw "Prototype start failed."
}
