param(
    [string]$CompilerDirectory = 'D:/app/latex/texlive/2026/bin/windows'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $projectRoot 'latex/1020'
$outputDirectory = Join-Path $projectRoot 'output/1020'
$pdfCompiler = Join-Path $CompilerDirectory 'pdflatex.exe'
$bibCompiler = Join-Path $CompilerDirectory 'bibtex.exe'

foreach ($executable in @($pdfCompiler, $bibCompiler)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Compiler not found: $executable"
    }
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$savedBibInputs = $env:BIBINPUTS
Push-Location -LiteralPath $sourceDirectory
try {
    & $pdfCompiler '-no-shell-escape' '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' "-output-directory=$outputDirectory" 'main.tex'
    if ($LASTEXITCODE -ne 0) { throw 'Initial LaTeX compilation failed. See output/1020/main.log.' }

    # BibTeX resolves the source bibliography while writing all generated files
    # beside the PDF rather than into the manuscript directory.
    $env:BIBINPUTS = $sourceDirectory + [IO.Path]::PathSeparator + $savedBibInputs
    Push-Location -LiteralPath $outputDirectory
    try {
        & $bibCompiler 'main'
        if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed. See output/1020/main.blg.' }
    } finally {
        Pop-Location
    }

    foreach ($pass in 1..2) {
        & $pdfCompiler '-no-shell-escape' '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' "-output-directory=$outputDirectory" 'main.tex'
        if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed on final pass $pass. See output/1020/main.log." }
    }
} finally {
    $env:BIBINPUTS = $savedBibInputs
    Pop-Location
}

Write-Output (Join-Path $outputDirectory 'main.pdf')
