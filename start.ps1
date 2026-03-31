param(
    [ValidateSet("hybrid", "bm25", "vector")]
    [string]$Mode = "hybrid",
    [ValidateSet("Default", "A", "B")]
    [string]$Setup = "Default",
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

function Invoke-PrototypeCommand {
    param(
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][ValidateSet("default", "a", "b")][string]$SetupName
    )

    $previousSetup = $env:CTI_RAG_SETUP
    try {
        if ($SetupName -eq "default") {
            Remove-Item Env:CTI_RAG_SETUP -ErrorAction SilentlyContinue
        } else {
            $env:CTI_RAG_SETUP = $SetupName
        }

        & $VenvPython @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "Prototype start failed."
        }
    }
    finally {
        if ([string]::IsNullOrEmpty($previousSetup)) {
            Remove-Item Env:CTI_RAG_SETUP -ErrorAction SilentlyContinue
        } else {
            $env:CTI_RAG_SETUP = $previousSetup
        }
    }
}

$setupName = $Setup.ToLowerInvariant()
if ($Question) {
    Invoke-PrototypeCommand -ArgumentList @("main.py", "query", $Question, "--mode", $Mode) -SetupName $setupName
} else {
    Invoke-PrototypeCommand -ArgumentList @("main.py", "interactive", "--mode", $Mode) -SetupName $setupName
}
