<#
.SYNOPSIS
    Demo Part 1 of 3 - MEKHLA - The problem, and M2 Data Engineering.
    Target runtime ~2:00.

.DESCRIPTION
    Nothing here trains a model or rebuilds the dataset. Every figure is read
    from artifacts the pipeline already produced, which keeps the segment to two
    minutes and means a slow stage cannot derail the recording.

    Run `defectvision data` at least once before recording so the artifacts exist.

.EXAMPLE
    .\scripts\demo\part1_mekhla.ps1
.EXAMPLE
    .\scripts\demo\part1_mekhla.ps1 -Pace 1.25   # slower, if it reads rushed
#>
[CmdletBinding()]
param([double]$Pace = 1.0)

$script:Pace = $Pace
. (Join-Path $PSScriptRoot 'demo_common.ps1')
Initialize-DemoConsole

$py = '.venv\Scripts\python.exe'

# ================================================================= 0:00
Show-Title -Part 'Part 1 of 3' -Presenter 'Mekhla' `
           -Topic 'The Problem, and M2 - Data Engineering' -Timecode '00:00 - 02:00'

# ================================================================= 0:10
Show-Caption -Heading 'THE PROBLEM' -Lines @(
    'A quality-assurance team needs to flag defective submersible-pump',
    'impellers from images taken on the production line -- and needs the',
    'system to keep working when the lighting, the camera, or the product',
    'mix changes.',
    '',
    'The interesting part is not the classifier. It is everything around it:',
    'keeping the data trustworthy, making runs reproducible, shipping the',
    'model safely, and knowing when it has stopped working.'
)

Show-Caption -Heading 'THE DATA' -Lines @(
    'Kaggle "Casting Product Image Data for Quality Inspection".',
    '7,348 real grayscale images at 300x300. Two classes: ok and defect.'
) -Hold 4.5

# ================================================================= 0:40
Show-Caption -Lines @(
    'These are the real images. Good parts on the top row, defective below.'
) -Hold 3
Show-Image 'reports/figures/raw_samples.png' -Hold 8

# ================================================================= 0:55
Show-Caption -Heading 'M2 - DATA VALIDATION' -Lines @(
    'Before any model is trained, the corpus passes seven quality gates.',
    'A per-file problem quarantines that file. A corpus-level problem',
    'aborts the whole run.'
)

Invoke-Demo -Command "& $py scripts\demo\show.py validation" -After 9

# ================================================================= 1:20
Show-Caption -Heading 'WHAT VALIDATION ACTUALLY FOUND' -Lines @(
    'These are real defects in the published dataset, not hypotheticals:',
    '',
    '     64 exact duplicate files   ->  dropped before splitting',
    '    412 near-duplicates         ->  GROUPED, not dropped'
)

Show-Point 'Near-duplicates are legitimate data - the same physical part, photographed twice.'

Show-Caption -Lines @(
    'But if copies of one part land on both sides of the train/test split,',
    'the test score measures memorisation, not generalisation.',
    '',
    'So they are detected by perceptual hash and forced into the SAME fold.',
    'This is the most common reason an image classifier scores 0.99 offline',
    'and then disappoints in production.'
)

# ================================================================= 1:45
Show-Caption -Heading 'THE VERSIONED DATASET' -Lines @(
    'The output is a manifest FILE, not a split computed at run time -- so it',
    'is hashable, reviewable in a diff, and still answerable months later.'
) -Hold 5

Invoke-Demo -Command "& $py scripts\demo\show.py dataset" -After 8

Show-Point 'Every fold holds 57.8% defect - stratified to a tenth of a percent.'

# ================================================================= 2:00
Show-Outro 'Next - Shruti: M3, Experiment Tracking and Model Selection'
