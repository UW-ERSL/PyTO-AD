param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Executable,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$executablePath = $null
$command = Get-Command $Executable -ErrorAction SilentlyContinue
if ($command) {
    $executablePath = $command.Source
}

if (-not $executablePath) {
    $binDirectories = @(
        "$env:LOCALAPPDATA\MiKTeX\miktex\bin\x64",
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
        "C:\Program Files\MiKTeX\miktex\bin\x64",
        "C:\Program Files (x86)\MiKTeX\miktex\bin\x64"
    )
    $executablePath = $binDirectories |
        ForEach-Object { Join-Path $_ $Executable } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
}

if (-not $executablePath) {
    Write-Error "$Executable was not found. Check the MiKTeX installation or add its bin folder to PATH."
    exit 1
}

& $executablePath @Arguments
exit $LASTEXITCODE
