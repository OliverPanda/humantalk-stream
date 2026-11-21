# Windows entry point; use the same planner and installer as Linux.
$ErrorActionPreference = 'Stop'
$installerArgs = $args
$sourceConfig = if ($env:FREEVIDEOAI_MIRROR_CONFIG) { $env:FREEVIDEOAI_MIRROR_CONFIG } else {
    Join-Path $PSScriptRoot 'config/install-mirrors.env'
}
$env:FREEVIDEOAI_MIRROR_CONFIG = $sourceConfig
if ($env:FREEVIDEOAI_DOWNLOAD_SOURCE -ne 'official') {
    foreach ($line in Get-Content -LiteralPath $sourceConfig) {
        $line = $line.Trim()
        if (!$line -or $line.StartsWith('#')) { continue }
        $parts = $line.Split('=', 2)
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw 'Invalid mirror configuration'
        }
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($parts[0], 'Process'))) {
            [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
        }
    }
}
$uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvBinary = $uvCommand.Source
} else {
    $uvBootstrap = Join-Path $env:LOCALAPPDATA 'FreeVideoAI\uv'
    & python -m pip install --upgrade --target $uvBootstrap uv
    if ($LASTEXITCODE -ne 0) { throw 'uv 安装失败，请先安装 Python 和 pip，或手动安装 uv。' }
    $uvBinary = & python -c "import sys; sys.path.insert(0, sys.argv[1]); import uv; print(uv.find_uv_bin())" $uvBootstrap
    if ($LASTEXITCODE -ne 0) { throw '无法定位 uv，请检查 Python/pip 环境。' }
}
$env:FREEVIDEOAI_UV = $uvBinary
& $uvBinary run --no-project --python 3.11 --with filelock (Join-Path $PSScriptRoot 'scripts/install.py') @installerArgs
exit $LASTEXITCODE
