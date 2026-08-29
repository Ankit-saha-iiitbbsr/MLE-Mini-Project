<#
.SYNOPSIS
    Demo Part 2 of 3 - SHRUTI - M3 Experiment tracking, model comparison,
    and gated promotion. Target runtime ~2:00.

.DESCRIPTION
    Training is NOT run live -- four arms take about 31 minutes on CPU. This
    segment shows the tracked results those runs produced.

    Before recording, start the MLflow UI in a separate terminal so it can be
    shown at the 0:20 mark:

        mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.EXAMPLE
    .\scripts\demo\part2_shruti.ps1
#>
[CmdletBinding()]
param(
    [double]$Pace = 1.0,
    [switch]$SkipMlflowUI   # pass if the UI is not running
)

$script:Pace = $Pace
. (Join-Path $PSScriptRoot 'demo_common.ps1')
Initialize-DemoConsole

$py = '.venv\Scripts\python.exe'

# ================================================================= 0:00
Show-Title -Part 'Part 2 of 3' -Presenter 'Shruti' `
           -Topic 'M3 - Experiments, Comparison and Promotion' -Timecode '02:00 - 04:00'

# ================================================================= 0:10
Show-Caption -Heading 'FOUR MODELS, ONE FAIR COMPARISON' -Lines @(
    'Every arm trains on the identical manifest -- same images, same folds --',
    'and each is recorded as a tracked MLflow run. The brief asks for two.',
    '',
    '   baseline_cnn          a CNN trained from scratch',
    '   resnet18              ImageNet transfer learning',
    '   mobilenet_v3_small    ImageNet transfer learning',
    '   logreg_hog            HOG + logistic regression  (classical control)'
)

Show-Point 'The classical control exists so "we used deep learning" is a measured decision, not an assumption.'

# ================================================================= 0:35
if (-not $SkipMlflowUI) {
    Show-Caption -Lines @(
        'All four runs, tracked in MLflow.'
    ) -Hold 2.5
    Start-Process 'http://localhost:5000'
    Wait-Beat 10
}

# ================================================================= 0:50
Show-Caption -Heading 'THE RESULTS' -Lines @(
    'Test split held out and evaluated exactly once. The operating threshold',
    'was tuned on validation data only -- never on test.'
) -Hold 4.5

Invoke-Demo -Command "& $py scripts\demo\show.py models" -After 10

Show-Caption -Lines @(
    'The classical control reaches 0.9609 F1. That is a genuine attempt, not',
    'a straw man -- and the ~3.6 point gap is the quantitative argument for',
    'using a convolutional network at all.'
)

# ================================================================= 1:20
Show-Caption -Heading 'WHICH MODEL WINS?' -Lines @(
    'The 0.25M-parameter CNN scored above the 11M-parameter ResNet.',
    'But is that difference real?'
) -Hold 4

Invoke-Demo -Command "& $py scripts\demo\show.py intervals" -After 9

Show-Point 'Statistically tied on accuracy. Not tied on cost - one is 6x faster.'

# ================================================================= 1:40
Show-Caption -Heading 'PROMOTION IS A RULE, NOT A JUDGEMENT CALL' -Lines @(
    '   1. GATES first    F1 >= 0.85, defect recall >= 0.85, p95 <= 250 ms.',
    '                     A model that is merely best-of-a-bad-field is',
    '                     never eligible.',
    '',
    '   2. RANK           by the selection metric.',
    '',
    '   3. TIE-BREAK      on latency, among candidates scoring inside the',
    '                     leader''s confidence interval.'
)

Show-Caption -Lines @(
    'So when accuracy is statistically indistinguishable, the cheaper model',
    'wins on merit -- and baseline_cnn was promoted by that rule, not by a',
    'human preferring it.'
) -Hold 6

# ================================================================= 1:55
Show-Caption -Heading 'REPRODUCIBILITY' -Lines @(
    'A run you cannot rebuild is a result you cannot defend.'
) -Hold 3

Invoke-Demo -Command "& $py scripts\demo\show.py runs" -After 9

# ================================================================= 2:00
Show-Outro 'Next - Ankit: M4 Serving, and M5 Monitoring and Drift'
