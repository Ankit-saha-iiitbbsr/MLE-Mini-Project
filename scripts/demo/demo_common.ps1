<#
.SYNOPSIS
    Shared presentation helpers for the recorded demo.

.DESCRIPTION
    The demo is recorded WITHOUT voiceover, so everything the viewer needs to
    understand must be on screen. These helpers render large caption banners and
    hold each step long enough to read comfortably, then run the command
    underneath it.

    Pacing assumes ~3.5 words/second of silent reading. Every Hold value below
    was chosen from the caption length, not guessed.

    Not intended to be run directly -- dot-source it from a part script.
#>

$ErrorActionPreference = 'Stop'

# Project root is two levels up from scripts/demo/
$script:DemoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:DemoPy = Join-Path $script:DemoRoot '.venv\Scripts\python.exe'

# Global pacing multiplier. Bump to 1.3 if the recording feels rushed on
# playback, or drop to 0.8 for a rehearsal pass.
if (-not $script:Pace) { $script:Pace = 1.0 }

function Wait-Beat {
    param([double]$Seconds = 1.0)
    Start-Sleep -Milliseconds ([int]($Seconds * 1000 * $script:Pace))
}

function Show-Title {
    <#  Full-screen title card that opens each part.  #>
    param(
        [Parameter(Mandatory)][string]$Part,
        [Parameter(Mandatory)][string]$Presenter,
        [Parameter(Mandatory)][string]$Topic,
        [string]$Timecode = ''
    )
    Clear-Host
    Write-Host ''
    Write-Host ''
    Write-Host ('  ' + ('=' * 74)) -ForegroundColor DarkCyan
    Write-Host ''
    Write-Host '   DefectVision  |  Image-Based Defect / Quality Classifier' -ForegroundColor Gray
    Write-Host '   Machine Learning Engineering (PCAM* ZC412)  -  EC-1 Mini-Project' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host "   $Part" -ForegroundColor Yellow
    Write-Host "   $Topic" -ForegroundColor White
    Write-Host ''
    Write-Host "   Presented by: $Presenter" -ForegroundColor Cyan
    if ($Timecode) { Write-Host "   $Timecode" -ForegroundColor DarkGray }
    Write-Host ''
    Write-Host ('  ' + ('=' * 74)) -ForegroundColor DarkCyan
    Wait-Beat 5
}

function Show-Caption {
    <#  The on-screen "narration". This replaces the voiceover, so hold long
        enough that a viewer can read it without pausing the video.  #>
    param(
        # AllowEmptyString: blank entries are used as deliberate spacing between
        # caption lines, and a mandatory [string[]] rejects them by default.
        [Parameter(Mandatory)][AllowEmptyString()][string[]]$Lines,
        [string]$Heading = '',
        [double]$Hold = 0
    )
    Write-Host ''
    Write-Host ('  ' + ('-' * 74)) -ForegroundColor DarkGray
    if ($Heading) {
        Write-Host "   $Heading" -ForegroundColor Yellow
        Write-Host ''
    }
    foreach ($line in $Lines) {
        Write-Host "   $line" -ForegroundColor White
    }
    Write-Host ('  ' + ('-' * 74)) -ForegroundColor DarkGray

    if ($Hold -le 0) {
        # ~3.5 words per second of silent reading, with a 2.5 s floor.
        $words = ($Lines -join ' ').Split(' ').Count
        $Hold = [Math]::Max(2.5, $words / 3.5)
    }
    Wait-Beat $Hold
}

function Show-Point {
    <#  A single highlighted takeaway -- use sparingly, for the line that
        matters most in the segment.  #>
    param([Parameter(Mandatory)][string]$Text, [double]$Hold = 0)
    Write-Host ''
    Write-Host "   >> $Text" -ForegroundColor Green
    Write-Host ''
    if ($Hold -le 0) { $Hold = [Math]::Max(3.0, $Text.Split(' ').Count / 3.5) }
    Wait-Beat $Hold
}

function Invoke-Demo {
    <#  Show the command, pause so the viewer registers it, run it, then hold
        on the output.  #>
    param(
        [Parameter(Mandatory)][string]$Command,
        [double]$Before = 2.0,
        [double]$After = 4.0
    )
    Write-Host ''
    Write-Host "   PS> $Command" -ForegroundColor Cyan
    Wait-Beat $Before
    Write-Host ''
    try {
        Invoke-Expression $Command | Out-Host
    } catch {
        Write-Host "   (command failed: $($_.Exception.Message))" -ForegroundColor Red
    }
    Wait-Beat $After
}

function Show-Image {
    <#  Open a figure in the default viewer so it appears in the recording.
        The viewer window must be closed manually, or left open while the
        narration continues underneath.  #>
    param([Parameter(Mandatory)][string]$RelPath, [double]$Hold = 7.0)
    $full = Join-Path $script:DemoRoot $RelPath
    if (Test-Path $full) {
        Write-Host ''
        Write-Host "   [opening $RelPath]" -ForegroundColor DarkGray
        Invoke-Item $full
        Wait-Beat $Hold
    } else {
        Write-Host "   [missing figure: $RelPath]" -ForegroundColor Red
    }
}

function Show-Json {
    <#  Pretty-print selected fields of a JSON report -- far more readable on
        camera than dumping the whole file.  #>
    param(
        [Parameter(Mandatory)][string]$RelPath,
        [Parameter(Mandatory)][string]$PythonExpr,
        [double]$Hold = 6.0
    )
    Push-Location $script:DemoRoot
    try {
        & $script:DemoPy -c $PythonExpr 2>&1 | Where-Object { $_ -notmatch 'INFO|WARNING' } | Out-Host
    } finally {
        Pop-Location
    }
    Wait-Beat $Hold
}

function Show-Outro {
    param([Parameter(Mandatory)][string]$NextUp)
    Write-Host ''
    Write-Host ('  ' + ('=' * 74)) -ForegroundColor DarkCyan
    Write-Host ''
    Write-Host "   $NextUp" -ForegroundColor Yellow
    Write-Host ''
    Write-Host ('  ' + ('=' * 74)) -ForegroundColor DarkCyan
    Wait-Beat 4
}

function Initialize-DemoConsole {
    <#  Large font and a fixed window size keep the recording legible at 1080p.
        Run this before recording, not during.  #>
    Clear-Host
    try {
        $host.UI.RawUI.WindowTitle = 'DefectVision - EC-1 Mini-Project Demo'
        $size = $host.UI.RawUI.BufferSize
        $size.Width = 110
        $size.Height = 3000
        $host.UI.RawUI.BufferSize = $size
        $win = $host.UI.RawUI.WindowSize
        $win.Width = 110
        $win.Height = 32
        $host.UI.RawUI.WindowSize = $win
    } catch {
        # Terminal hosts that do not support resizing -- harmless.
    }
    Set-Location $script:DemoRoot
}
