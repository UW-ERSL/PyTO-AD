param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $LatexmkArguments
)

# MiKTeX's latexmk executable is a Perl script wrapper. Ensure Perl is visible
# even when VS Code was launched before Strawberry Perl was added to PATH.
$perlDirectories = @(
    "C:\\Strawberry\\perl\\bin",
    "C:\\Strawberry\\c\\bin"
) | Where-Object { Test-Path (Join-Path $_ "perl.exe") -or Test-Path $_ }
if ($perlDirectories) {
    $env:Path = (($perlDirectories -join ";") + ";" + $env:Path)
}

$latexmkPath = $null
$command = Get-Command latexmk.exe -ErrorAction SilentlyContinue
if ($command) {
    $latexmkPath = $command.Source
}

if (-not $latexmkPath) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\latexmk.exe",
        "C:\Program Files\MiKTeX\miktex\bin\x64\latexmk.exe",
        "C:\Program Files (x86)\MiKTeX\miktex\bin\x64\latexmk.exe"
    )
    $latexmkPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $latexmkPath) {
    Write-Error 'latexmk.exe was not found. Install the latexmk package in MiKTeX Console, then reload VS Code.'
    exit 1
}

& $latexmkPath @LatexmkArguments
exit $LASTEXITCODE
