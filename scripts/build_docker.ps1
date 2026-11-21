param([ValidateSet('auto', 'local', 'cpu', 'gpu', 'all')][string]$Target = 'auto')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($Target -eq 'auto') {
    $Target = 'local'
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $memory = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $memory) {
            $firstMemory = @($memory)[0].Trim()
            if ($firstMemory -match '^\d+$' -and [int]$firstMemory -ge 6144) { $Target = 'gpu' }
        }
    }
}
$buildArguments = @()
foreach ($key in @('FREEVIDEOAI_DOWNLOAD_SOURCE', 'PIP_INDEX_URL', 'UV_INDEX_URL',
    'HF_ENDPOINT', 'UV_PYTHON_INSTALL_MIRROR', 'FREEVIDEOAI_TORCH_INDEX',
    'FREEVIDEOAI_GITHUB_MIRROR', 'FREEVIDEOAI_APT_MIRROR', 'FREEVIDEOAI_MMCV_INDEX',
    'PLAYWRIGHT_DOWNLOAD_HOST')) {
    $value = [Environment]::GetEnvironmentVariable($key, 'Process')
    if ($value) { $buildArguments += @('--build-arg', "$key=$value") }
}
docker build @buildArguments -f (Join-Path $projectRoot 'Dockerfile') -t freevideoai:web $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Web image build failed' }
if ($Target -in @('gpu', 'all')) {
    docker build @buildArguments -f (Join-Path $projectRoot 'docker/Dockerfile.gpu') -t freevideoai:gpu $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'GPU image build failed' }
    Write-Host '启动：docker compose -f compose.yaml -f compose.gpu.yaml up -d'
} else {
    Write-Host '启动：docker compose up -d'
}
