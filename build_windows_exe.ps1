$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

Write-Host "Installing application build dependencies..."
py -3 -m pip install -e ".[app]"

Write-Host "Building the Windows executable..."
py -3 -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name FourWheelVehicleSimulation `
    --paths src `
    --collect-data matplotlib `
    --distpath release `
    --workpath build\pyinstaller `
    --specpath build\pyinstaller `
    examples\gui_launcher.py

$executable = Join-Path $projectRoot "release\FourWheelVehicleSimulation.exe"
Write-Host "Build complete: $executable"
