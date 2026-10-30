param(
    [string]$Compiler = 'D:/app/latex/texlive/2026/bin/windows/pdflatex.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $projectRoot 'latex/template/Conference-LaTeX-template_10-17-19'
$outputDirectory = Join-Path $projectRoot 'output/template'

if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw "LaTeX compiler not found: $Compiler"
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
Push-Location -LiteralPath $sourceDirectory
try {
    foreach ($pass in 1..2) {
        & $Compiler '-no-shell-escape' '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' "-output-directory=$outputDirectory" 'conference_101719.tex'
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX compilation failed on pass $pass. See output/template/conference_101719.log."
        }
    }
} finally {
    Pop-Location
}

Write-Output (Join-Path $outputDirectory 'conference_101719.pdf')
