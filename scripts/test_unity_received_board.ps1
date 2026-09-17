# PowerShell 7: Unityに依存しない実際のC#検証器に、modelの全局面を通す。
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$sourceRoot = Join-Path $repoRoot 'unity/KokoroShogi/Assets/_Project/Scripts'
$sources = @('Net/Messages.cs', 'Core/BoardState.cs', 'Core/ReceivedBoardSnapshot.cs')
Add-Type -Path ($sources | ForEach-Object { Join-Path $sourceRoot $_ })
$options = [System.Text.Json.JsonSerializerOptions]::new()
$options.IncludeFields = $true

function Read-State([string] $json) {
    return [System.Text.Json.JsonSerializer]::Deserialize($json, [KokoroShogi.Net.StateUpdate], $options)
}
function Assert-Rejected([KokoroShogi.Net.StateUpdate] $state) {
    $rejected = $false
    try { [KokoroShogi.Core.ReceivedBoardSnapshot]::new($state) | Out-Null }
    catch { $rejected = $true }
    if (-not $rejected) { throw 'Invalid snapshot was accepted.' }
}

$states = 0
$captures = 0
$promotions = 0
$drops = 0
$firstJson = $null
$files = @(Get-ChildItem (Join-Path $repoRoot 'sample_data/model') -Filter 'game*.jsonl')
foreach ($file in $files) {
    foreach ($line in [System.IO.File]::ReadLines($file.FullName)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $state = Read-State $line
        if ($state.type -ne 'state_update') { continue }
        if (-not $firstJson) { $firstJson = $line }
        try { $snapshot = [KokoroShogi.Core.ReceivedBoardSnapshot]::new($state) }
        catch { throw "$($file.Name) ply=$($state.ply): $_" }
        $expectedPlayer = if ($state.sfen.Split(' ')[1] -eq 'b') { 0 } else { 1 }
        if ($snapshot.PlayerToMove -ne $expectedPlayer) { throw 'Wrong side to move.' }
        $states++
        if ($state.last_move.capture) { $captures++ }
        if ($state.last_move.promote) { $promotions++ }
        if ($state.last_move.drop) { $drops++ }
    }
}
if ($states -eq 0 -or $captures -eq 0 -or $promotions -eq 0 -or $drops -eq 0) {
    throw 'Sample coverage is incomplete.'
}
$state = Read-State $firstJson
$state.pieces[1].piece_id = $state.pieces[0].piece_id
Assert-Rejected $state
$state = Read-State $firstJson
$state.pieces[0].square = '00'
Assert-Rejected $state
$state = Read-State $firstJson
$state.pieces[0].owner = 2
Assert-Rejected $state
$state = Read-State $firstJson
$state.pieces[0].species = 'UNKNOWN'
Assert-Rejected $state
$state = Read-State $firstJson
$state.sfen = $state.sfen.Replace(' b - ', ' b P ')
Assert-Rejected $state
$state = Read-State $firstJson
$state.pieces.RemoveAt(0)
Assert-Rejected $state
foreach ($badBoard in @('+9/9/9/9/9/9/9/9/9', '++P8/9/9/9/9/9/9/9/9', '09/9/9/9/9/9/9/9/9')) {
    $state = Read-State $firstJson
    $state.sfen = "$badBoard b - 1"
    Assert-Rejected $state
}
Write-Output "PASS: $($files.Count) files, $states states, $captures captures, $promotions promotions, $drops drops; 9 invalid snapshots rejected."
