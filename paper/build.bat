@echo off
REM Double-click this file to compile main.tex -> main.pdf (MiKTeX, no Perl needed).
cd /d "%~dp0"
set "BIN=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"

echo [1/4] pdflatex ...
"%BIN%\pdflatex.exe" -interaction=nonstopmode main.tex >nul
echo [2/4] bibtex ...
"%BIN%\bibtex.exe" main >nul
echo [3/4] pdflatex ...
"%BIN%\pdflatex.exe" -interaction=nonstopmode main.tex >nul
echo [4/4] pdflatex ...
"%BIN%\pdflatex.exe" -interaction=nonstopmode main.tex >nul

if exist main.pdf (
  echo.
  echo BUILD OK - opening main.pdf
  start "" "main.pdf"
) else (
  echo.
  echo BUILD FAILED - open main.log to see the error
)
echo.
pause
