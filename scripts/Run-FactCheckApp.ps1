param(
    [string]$ProjectRoot = "D:\FactCheckPipeline",
    [string]$CondaEnv = "vector_db",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501,
    [switch]$KillExisting,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[FactCheck] $Message"
}

function Test-Http {
    param(
        [string]$Url,
        [int]$TimeoutSec = 5
    )
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSec | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-PortProcess {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $connections) {
        return
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
    foreach ($processId in $processIds) {
        if ($processId -and $processId -ne 0) {
            Write-Step "Stopping process $processId on port $Port"
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Assert-PortFree {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connections) {
        throw "Port $Port is already in use. Re-run with -KillExisting or stop the process manually."
    }
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$TimeoutSec = 600
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http -Url $Url -TimeoutSec 5) {
            return
        }
        Start-Sleep -Seconds 3
    }
    throw "Timed out waiting for $Url"
}

if (-not (Test-Path $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

$BackendRoot = Join-Path $ProjectRoot "src\backend"
$FrontendApp = Join-Path $ProjectRoot "src\frontend\app.py"
$LogDir = Join-Path $ProjectRoot "logs"
$BackendOut = Join-Path $LogDir "backend_stdout.log"
$BackendErr = Join-Path $LogDir "backend_stderr.log"
$FrontendOut = Join-Path $LogDir "frontend_stdout.log"
$FrontendErr = Join-Path $LogDir "frontend_stderr.log"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

Write-Step "Checking Qdrant at http://127.0.0.1:6333"
if (-not (Test-Http -Url "http://127.0.0.1:6333/collections")) {
    throw "Qdrant is not reachable at http://127.0.0.1:6333. Start Qdrant first."
}

if ($KillExisting) {
    Stop-PortProcess -Port $BackendPort
    Stop-PortProcess -Port $FrontendPort
    Start-Sleep -Seconds 1
} else {
    Assert-PortFree -Port $BackendPort
    Assert-PortFree -Port $FrontendPort
}

Write-Step "Starting FastAPI backend on port $BackendPort using conda env '$CondaEnv'"
$BackendCommand = "cd /d `"$BackendRoot`" && conda run --no-capture-output -n $CondaEnv python -m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort"
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", $BackendCommand `
    -RedirectStandardOutput $BackendOut `
    -RedirectStandardError $BackendErr `
    -WindowStyle Hidden

Write-Step "Waiting for backend. First startup may load models and take several minutes."
Wait-Http -Url "http://127.0.0.1:$BackendPort/api/v1/pipeline/debug/runtime" -TimeoutSec 900

Write-Step "Starting Streamlit frontend on port $FrontendPort"
$FrontendCommand = "cd /d `"$ProjectRoot`" && conda run --no-capture-output -n $CondaEnv streamlit run `"$FrontendApp`" --server.address 127.0.0.1 --server.port $FrontendPort"
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", $FrontendCommand `
    -RedirectStandardOutput $FrontendOut `
    -RedirectStandardError $FrontendErr `
    -WindowStyle Hidden

Write-Step "Waiting for frontend"
Wait-Http -Url "http://127.0.0.1:$FrontendPort" -TimeoutSec 120

$AppUrl = "http://127.0.0.1:$FrontendPort"
Write-Step "App is running: $AppUrl"
Write-Step "Backend logs: $BackendOut"
Write-Step "Frontend logs: $FrontendOut"

if ($OpenBrowser) {
    Start-Process $AppUrl
}

