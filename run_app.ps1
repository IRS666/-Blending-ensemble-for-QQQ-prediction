$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { throw "Missing .venv. Create it with: py -m venv .venv" }
Set-Location $projectRoot
& $pythonExe -m streamlit run app.py --server.address localhost --server.port 8510
