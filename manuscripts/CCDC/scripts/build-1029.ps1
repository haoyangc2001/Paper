param(
    [string]$CompilerDirectory = 'D:/app/latex/texlive/2026/bin/windows',
    [string]$Python = 'D:/codesoftware/Miniconda/python.exe',
    [string]$CompileHelper = 'C:/Users/smile/.codex/skills/latex-paper-en/scripts/compile.py'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $projectRoot 'latex/1029'
$outputDirectory = Join-Path $projectRoot 'output/1029'
foreach ($required in @($Python, $CompileHelper, (Join-Path $CompilerDirectory 'latexmk.exe'), (Join-Path $CompilerDirectory 'pdflatex.exe'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing build dependency: $required" }
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$savedPath = $env:PATH
$savedBibInputs = $env:BIBINPUTS
try {
    $env:PATH = $CompilerDirectory + [IO.Path]::PathSeparator + $savedPath
    $env:BIBINPUTS = $sourceDirectory + [IO.Path]::PathSeparator + $savedBibInputs
    & $Python -B $CompileHelper (Join-Path $sourceDirectory 'main.tex') --compiler pdflatex --outdir $outputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed. See output/1029/main.log.' }
} finally {
    $env:PATH = $savedPath
    $env:BIBINPUTS = $savedBibInputs
}
Write-Output (Join-Path $outputDirectory 'main.pdf')
