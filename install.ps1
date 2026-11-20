# Windows entry point; use the same planner and installer as Linux.
$ErrorActionPreference = 'Stop'
$installerArgs = $args
$uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvBinary = $uvCommand.Source
} else {
    & python -m pip install --user uv
    if ($LASTEXITCODE -ne 0) { throw 'uv 安装失败，请先安装 Python 和 pip，或手动安装 uv。' }
    $uvBinary = & python -c "import uv; print(uv.find_uv_bin())"
    if ($LASTEXITCODE -ne 0) { throw '无法定位 uv，请检查 Python/pip 环境。' }
}
$env:FREEVIDEOAI_UV = $uvBinary
& $uvBinary run --no-project --python 3.11 --with filelock (Join-Path $PSScriptRoot 'scripts/install.py') @installerArgs
exit $LASTEXITCODE
