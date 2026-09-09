@echo off
setlocal DisableDelayedExpansion
title Qwen Flash-Next - Chat locale
echo Qwen Flash-Next - Q4, MTP e contesto 24K
echo Scrivi la prima domanda quando compare Tu: e premi Invio.
echo Il primo caricamento richiede circa 14 minuti; i turni successivi riusano il modello.
echo Comandi: /reset per una nuova conversazione, /exit per uscire.
echo.
pushd "%~dp0"
wsl.exe -d Ubuntu -- qwen/build/venv/bin/python -u qwen/chat.py %*
set "qwen_exit_code=%errorlevel%"
popd
if not "%qwen_exit_code%"=="0" (
    echo.
    echo Avvio o esecuzione di Qwen non riusciti. Codice: %qwen_exit_code%
    pause
)
exit /b %qwen_exit_code%
