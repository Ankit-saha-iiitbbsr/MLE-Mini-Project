<#
.SYNOPSIS
    Demo Part 3 of 3 - ANKIT - M4 Serving and M5 Monitoring / Drift / Retraining.
    Target runtime ~2:20.

.DESCRIPTION
    This is the only part that hits a live service. Start the API in a separate
    terminal BEFORE recording and leave it running:

        defectvision serve

    The script checks readiness first and will tell you if it is not up. The
    drift and monitoring figures are read from artifacts, not regenerated --
    `simulate-drift` takes about five minutes.

.EXAMPLE
    .\scripts\demo\part3_ankit.ps1
#>
[CmdletBinding()]
param([double]$Pace = 1.0)

$script:Pace = $Pace
. (Join-Path $PSScriptRoot 'demo_common.ps1')
Initialize-DemoConsole

$py = '.venv\Scripts\python.exe'
$api = 'http://localhost:8000'

# --- pre-flight: fail loudly BEFORE recording rather than mid-take ---------
try {
    $ready = Invoke-RestMethod "$api/readyz" -TimeoutSec 5
    if (-not $ready.model_loaded) { throw 'model not loaded' }
} catch {
    Write-Host ''
    Write-Host '  The API is not running. Start it in another terminal first:' -ForegroundColor Red
    Write-Host '      defectvision serve' -ForegroundColor Yellow
    Write-Host ''
    exit 1
}

# ================================================================= 0:00
Show-Title -Part 'Part 3 of 3' -Presenter 'Ankit' `
           -Topic 'M4 - Serving  |  M5 - Monitoring and Drift' -Timecode '04:00 - 06:20'

# ================================================================= 0:10
Show-Caption -Heading 'M4 - THE DEPLOYED SERVICE' -Lines @(
    'The promoted model is served behind a FastAPI application. Liveness and',
    'readiness are SEPARATE endpoints, on purpose.'
) -Hold 4.5

Invoke-Demo -Command "Invoke-RestMethod $api/readyz | Format-List" -After 5

Show-Caption -Lines @(
    'A container whose model failed to load is alive but must not receive',
    'traffic. Collapse the two and you either restart-loop a healthy process,',
    'or route real requests to a broken one.'
)

# ================================================================= 0:35
Show-Caption -Heading 'A REAL PREDICTION' -Lines @(
    'A known-defective casting, posted to the live service.'
) -Hold 3

# ConvertFrom-Json/ConvertTo-Json rather than piping to `python -m json.tool`:
# PowerShell prefixes piped native-command output with a UTF-8 BOM, which the
# Python JSON decoder rejects.
Invoke-Demo -Command @"
curl.exe -s -F "file=@data/raw/defect/test_cast_def_0_1059.jpeg" $api/predict | ConvertFrom-Json | ConvertTo-Json -Depth 5
"@ -After 10

Show-Caption -Lines @(
    'Note what comes back besides the label:',
    '',
    '   decision      routes borderline scores to a HUMAN, not an auto-action',
    '   threshold     travels with the answer, so the decision is auditable',
    '   image_stats   the drift features, computed at request time',
    '   request_id    ties this response to its row in the prediction log'
)

# ================================================================= 1:00
Show-Caption -Heading 'MALFORMED INPUT' -Lines @(
    'A 5xx in the logs should mean "we have a bug". That signal is worthless',
    'if a corrupt upload also produces one.'
) -Hold 4.5

# The status code is appended with -w on one line; embedding a newline in the
# format string breaks how the command renders on screen.
'not an image at all' | Out-File -Encoding ascii bad.png
Invoke-Demo -Command @"
curl.exe -s -w "   <-- HTTP %{http_code}" -F "file=@bad.png" $api/predict
"@ -After 8

Show-Point '400 with an actionable hint - not a 500.'
Remove-Item bad.png -ErrorAction SilentlyContinue

Invoke-Demo -Command "& $py scripts\demo\show.py benchmark" -After 8

# ================================================================= 1:25
Show-Caption -Heading 'M5 - DOES IT STILL WORK TOMORROW?' -Lines @(
    'Eight scenarios pushed through the deployed model. One uncorrupted',
    'control, five physically-motivated corruptions, and one REAL shift.'
) -Hold 5

Invoke-Demo -Command "& $py scripts\demo\show.py drift" -After 11

Show-Caption -Lines @(
    'The baseline control sits at 0.051 PSI -- that is the noise floor, so',
    'every other row is read against it rather than against zero.',
    '',
    'And real_camera_upgrade is NOT simulated: a second capture of the same',
    'line by a different camera, held back from training so the detectors',
    'face a shift no corruption operator was tuned against. It was caught.'
)

# ================================================================= 1:50
Show-Caption -Heading 'THE HEADLINE FINDING' -Lines @(
    'Look at what happens under dim lighting.'
) -Hold 3

Invoke-Demo -Command "& $py scripts\demo\show.py silent" -After 12

Show-Point 'A silent failure. Confidence monitoring alone would have missed six of seven.'

Show-Caption -Lines @(
    'The pattern of WHICH features drift also names the fault -- defocus',
    'fires the focus statistic while leaving brightness at the noise floor.',
    'The alert says which part of the capture rig to go and inspect.'
) -Hold 6
Show-Image 'reports/figures/monitoring/psi_heatmap.png' -Hold 9

# ================================================================= 2:10
Show-Caption -Heading 'SO DO WE RETRAIN?' -Lines @(
    'A naive rule would have fired immediately. This one is a policy:',
    'persistence across windows, a cooldown, and enough newly labelled data.'
) -Hold 5

Invoke-Demo -Command "& $py scripts\demo\show.py retrain" -After 12

Show-Point 'Drift does not automatically mean retrain. Retraining on a broken camera bakes the fault in.'

# ================================================================= 2:20
Show-Caption -Heading 'THE LOOP' -Lines @(
    'Data  ->  tracked experiments  ->  gated promotion  ->  serving',
    '      ->  monitoring  ->  back to data.',
    '',
    'That loop is what makes it a system rather than a model.'
) -Hold 7

Show-Outro 'DefectVision  -  Mekhla, Shruti, Ankit  -  Thank you'
