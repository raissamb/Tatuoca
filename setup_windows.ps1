# setup_windows.ps1
Write-Host "=== [1/4] Configurando Ambiente Virtual Python ===" -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

Write-Host "=== [2/4] Instalando dependências ===" -ForegroundColor Cyan
pip install --upgrade pip
pip install setuptools cython numpy opencv-python pandas matplotlib scipy

Write-Host "=== [3/4] Compilando lpmalgos (C++) via CMake ===" -ForegroundColor Cyan
cd sources/lpmalgos
if (Test-Path "build") { Remove-Item -Recurse -Force build }
cmake -B build
cmake --build build --config Release
cd ../..
if (-Not (Test-Path "lpmalgos")) { New-Item -ItemType Directory -Path "lpmalgos" }
Get-ChildItem -Path "sources\lpmalgos" -Filter "lpmalgos*.pyd" -Recurse | Copy-Item -Destination "lpmalgos\" -Force

Write-Host "=== [4/4] Compilando skeleton (Cython) ===" -ForegroundColor Cyan
cd sources/skeleton
if (Test-Path "build") { Remove-Item -Recurse -Force build }
python setup.py build_ext --inplace
cd ../..
if (-Not (Test-Path "skeleton")) { New-Item -ItemType Directory -Path "skeleton" }
Get-ChildItem -Path "sources\skeleton" -Filter "skeleton*.pyd" -Recurse | Copy-Item -Destination "skeleton\" -Force

Write-Host "=== Concluido! Testando importacoes ===" -ForegroundColor Green
python -c "import lpmalgos; import skeleton; print('-> Tudo instalado com sucesso no Windows!')"