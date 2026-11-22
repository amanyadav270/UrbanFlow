param(
    [switch]$Restart,
    [switch]$NoBrowser,
    [switch]$InstallDependencies,
    [switch]$CheckOnly
)

$launchArgs = @()
if ($Restart) { $launchArgs += '--restart' }
if ($NoBrowser) { $launchArgs += '--no-browser' }
if ($InstallDependencies) { $launchArgs += '--install-dependencies' }
if ($CheckOnly) { $launchArgs += '--check-only' }

& (Join-Path $PSScriptRoot 'START_URBANFLOW.bat') @launchArgs
exit $LASTEXITCODE
