param(
    [int]$MispMaxEvents = 300,
    [ValidateSet("Default", "A", "B", "Both")][string]$Setup = "Default",
    [switch]$SkipDownload,
    [switch]$SkipIndex,
    [switch]$SkipWarmup,
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3.12")
    }
    if (Get-Command python3.12 -ErrorAction SilentlyContinue) {
        return @("python3.12")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python 3.12 was not found. Install Python 3.12 and retry."
}

function Invoke-CommandChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($ArgumentList -join ' ')"
    }
}

function Invoke-IndexBuild {
    param(
        [Parameter(Mandatory = $true)][string]$VenvPythonPath,
        [Parameter(Mandatory = $true)][ValidateSet("default", "a", "b")][string]$SetupName
    )

    $previousSetup = $env:CTI_RAG_SETUP
    try {
        if ($SetupName -eq "default") {
            Remove-Item Env:CTI_RAG_SETUP -ErrorAction SilentlyContinue
            Write-Host "Building indexes for default setup..."
        } else {
            $env:CTI_RAG_SETUP = $SetupName
            Write-Host "Building indexes for setup $($SetupName.ToUpper())..."
        }

        Invoke-CommandChecked -FilePath $VenvPythonPath -ArgumentList @("main.py", "index", "--clear")
    }
    finally {
        if ([string]::IsNullOrEmpty($previousSetup)) {
            Remove-Item Env:CTI_RAG_SETUP -ErrorAction SilentlyContinue
        } else {
            $env:CTI_RAG_SETUP = $previousSetup
        }
    }
}

$PythonCommand = Resolve-PythonCommand
$PythonExecutable = $PythonCommand[0]
$PythonArguments = @()
if ($PythonCommand.Length -gt 1) {
    $PythonArguments = $PythonCommand[1..($PythonCommand.Length - 1)]
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $RepoRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Invoke-CommandChecked -FilePath $PythonExecutable -ArgumentList ($PythonArguments + @("-m", "venv", ".venv"))
}

Invoke-CommandChecked -FilePath $VenvPip -ArgumentList @("install", "-r", "requirements.txt")

$LlmModel = (& $VenvPython -c "from src.cti_rag.utils.config import load_config; print(load_config()['llm']['model_name'])").Trim()
if (-not $SkipOllama) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Invoke-CommandChecked -FilePath "ollama" -ArgumentList @("pull", $LlmModel)
    } else {
        Write-Warning "Ollama was not found on PATH. Install Ollama manually and pull '$LlmModel' before running the prototype."
    }
}

if (-not $SkipWarmup) {
    Invoke-CommandChecked -FilePath $VenvPython -ArgumentList @(
        "-c",
        "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction; from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever; from src.cti_rag.utils.config import load_config; config = load_config(); emb = config['embedding']; reranker = config['retrieval']['reranker']['model_name']; SentenceTransformerEmbeddingFunction(model_name=emb['model_name'], device=emb['device'])(['warmup']); HybridRetriever._ensure_local_reranker_snapshot(reranker); print(f'Cached embedding model: {emb[""model_name""]}'); print(f'Cached reranker model: {reranker}')"
    )
}

if (-not $SkipDownload) {
    Invoke-CommandChecked -FilePath $VenvPython -ArgumentList @("main.py", "download", "--source", "cisa_kev")
    Invoke-CommandChecked -FilePath $VenvPython -ArgumentList @("main.py", "download", "--source", "cisa_advisories")
    Invoke-CommandChecked -FilePath $VenvPython -ArgumentList @("main.py", "download", "--source", "nvd")
    Invoke-CommandChecked -FilePath $VenvPython -ArgumentList @("main.py", "download", "--source", "misp", "--misp-max-events", $MispMaxEvents.ToString())
}

if (-not $SkipIndex) {
    switch ($Setup.ToLowerInvariant()) {
        "default" { Invoke-IndexBuild -VenvPythonPath $VenvPython -SetupName "default" }
        "a" { Invoke-IndexBuild -VenvPythonPath $VenvPython -SetupName "a" }
        "b" { Invoke-IndexBuild -VenvPythonPath $VenvPython -SetupName "b" }
        "both" {
            Invoke-IndexBuild -VenvPythonPath $VenvPython -SetupName "a"
            Invoke-IndexBuild -VenvPythonPath $VenvPython -SetupName "b"
        }
    }
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start the prototype with:"
Write-Host "  .\start.ps1"
