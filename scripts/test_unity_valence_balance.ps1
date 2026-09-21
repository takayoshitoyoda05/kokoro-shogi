# Unity非依存の本番C#計算を直接検証する。
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$sourceRoot = Join-Path $repoRoot 'unity/KokoroShogi/Assets/_Project/Scripts'
Add-Type -Path @((Join-Path $sourceRoot 'Net/Messages.cs'), (Join-Path $sourceRoot 'Core/TeamValenceBalance.cs'))

function New-Piece([int] $owner, [float] $valence, [string] $square = '11') {
    $piece = [KokoroShogi.Net.PieceInfo]::new()
    $piece.owner = $owner
    $piece.square = $square
    $piece.mood = [KokoroShogi.Net.Mood]::new()
    $piece.mood.valence = $valence
    return $piece
}
function Assert-Ratio([string] $label, [KokoroShogi.Net.PieceInfo[]] $pieces, [float] $expected) {
    $actual = [KokoroShogi.Core.TeamValenceBalance]::CalculateFirstPlayerRatio($pieces)
    if ([Math]::Abs($actual - $expected) -gt 0.00001) { throw "$label expected=$expected actual=$actual" }
    Write-Output "PASS: $label"
}

Assert-Ratio '3:1 totals' @((New-Piece 0 1), (New-Piece 0 1), (New-Piece 0 1), (New-Piece 1 1)) 0.75
Assert-Ratio 'Equal totals' @((New-Piece 0 0.2), (New-Piece 1 0.2)) 0.5
Assert-Ratio 'Zero totals' @((New-Piece 0 0), (New-Piece 1 0)) 0.5
Assert-Ratio 'No snapshot' $null 0.5
Assert-Ratio 'Only second player positive' @((New-Piece 0 -0.5), (New-Piece 1 0.5)) 0
Assert-Ratio 'Both totals negative' @((New-Piece 0 -0.5), (New-Piece 1 -0.2)) 0.5
Assert-Ratio 'Sum before clamping' @((New-Piece 0 0.8), (New-Piece 0 -0.6), (New-Piece 1 0.2)) 0.5
$captured = New-Piece 0 0.6 'hand'
$boardPiece = New-Piece 1 0.2
Assert-Ratio 'Hand piece included' @($captured, $boardPiece) 0.75
$captured.owner = 1
Assert-Ratio 'Ownership change' @($captured, $boardPiece) 0
Assert-Ratio 'Invalid data ignored' @($null, [KokoroShogi.Net.PieceInfo]::new(), (New-Piece 0 ([float]::NaN)), (New-Piece 0 ([float]::PositiveInfinity)), (New-Piece 2 1), (New-Piece 1 0.4)) 0
