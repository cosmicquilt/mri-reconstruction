# Single-command runner for Windows / PowerShell.
#   .\run.ps1                 # smoke test (NumPy only)
#   .\run.ps1 train --config configs/synthetic.yaml
#   .\run.ps1 eval  --config configs/eval.yaml
#   .\run.ps1 figures --config configs/eval.yaml
# Sets PYTHONPATH so it works without `pip install -e .`.
param(
    [Parameter(Position = 0)]
    [string]$Command = "smoke",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
python -m mri_recon.cli $Command @Rest
exit $LASTEXITCODE
