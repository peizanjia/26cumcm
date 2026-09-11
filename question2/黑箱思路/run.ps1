param(
    [ValidateSet('train','evaluate','candidates','plot','test','all')]
    [string]$Task = 'all',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
# Avoid loading a second Intel OpenMP runtime through Anaconda MKL alongside
# PyTorch. This selects MKL's supported sequential backend, not duplicate-lib bypass.
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:NUMBA_NUM_THREADS = '12'
$repositoryPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Push-Location -LiteralPath $repositoryPath
try {
    if ($Task -in @('train','all')) {
        & $Python -m question2.黑箱思路.train --steps 6000 --states 32768 --posterior 128 --batch 256 --samples 32 --models 4 --validate-every 300
        if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    }
    if ($Task -in @('evaluate','all')) {
        & $Python -m question2.黑箱思路.evaluate
        if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    }
    if ($Task -in @('candidates','all')) {
        & $Python -m question2.黑箱思路.candidates --examples
        if ($LASTEXITCODE -ne 0) { throw 'Candidate evaluation failed' }
    }
    if ($Task -in @('plot','all')) {
        & $Python -m question2.黑箱思路.plot_results
        if ($LASTEXITCODE -ne 0) { throw 'Plotting failed' }
    }
    if ($Task -in @('test','all')) {
        & $Python -m question2.黑箱思路.verify
        if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    }
} finally {
    Pop-Location
}
