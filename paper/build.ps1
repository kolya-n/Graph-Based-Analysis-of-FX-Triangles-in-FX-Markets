# Build the thesis PDF with MiKTeX (natbib + bibtex; no Perl/latexmk needed).
# Usage: open PowerShell in this folder and run  .\build.ps1
$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
Set-Location -Path $PSScriptRoot

& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex   # 1: writes .aux
& "$bin\bibtex.exe"   main                                # 2: resolves bibliography
& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex   # 3: pulls in citations
& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex   # 4: settles cross-refs

if (Test-Path main.pdf) {
    Write-Host "OK: main.pdf built ($([math]::Round((Get-Item main.pdf).Length/1KB,1)) KB)" -ForegroundColor Green
} else {
    Write-Host "FAILED: no main.pdf - check main.log" -ForegroundColor Red
}
