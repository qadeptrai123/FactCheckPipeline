param(
    [int[]]$Ports = @(8000, 8501)
)

$ErrorActionPreference = "Stop"

foreach ($Port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "Port $Port is free"
        continue
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
    foreach ($processId in $processIds) {
        if (-not $processId -or $processId -eq 0) {
            continue
        }

        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping $($process.ProcessName) PID $processId on port $Port"
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

