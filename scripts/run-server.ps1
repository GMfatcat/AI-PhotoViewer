<#
.SYNOPSIS
  Launch the AI-PhotoViewer web server (Windows / PowerShell).

.EXAMPLE
  scripts\run-server.ps1
  scripts\run-server.ps1 -Model C:\models\siglip2-so400m -Port 8080 -BindHost 0.0.0.0

.NOTES
  Model dir can also be set via the SIGLIP_MODEL environment variable.
  An empty photos.db is created automatically on first run.
#>
param(
  [string]$Db,
  [string]$Model,
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8000
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot          # repo root (parent of scripts\)
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Venv)) {
  Write-Error "venv python not found: $Venv`nCreate it first:  uv venv  (then install deps)"
  exit 1
}

if (-not $Db)    { $Db = Join-Path $Root "photos.db" }
if (-not $Model) {
  if ($env:SIGLIP_MODEL) { $Model = $env:SIGLIP_MODEL }
  else { $Model = Join-Path $Root "..\models\siglip2-so400m" }
}

Write-Host "DB:    $Db"
Write-Host "Model: $Model"
Write-Host "URL:   http://$BindHost`:$Port"
& $Venv (Join-Path $Root "web_demo\main.py") --db $Db --model $Model --host $BindHost --port $Port
