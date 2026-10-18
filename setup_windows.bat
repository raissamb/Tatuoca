@echo off
echo === [1/4] Criando Ambiente Virtual Python ===
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo === [2/4] Instalando dependencias ===
python -m pip install --upgrade pip
pip install setuptools cython numpy opencv-python pandas matplotlib scipy

echo === [3/4] Compilando lpmalgos (C++) ===
cd sources\lpmalgos
if exist build rd /s /q build
cmake -B build
cmake --build build --config Release
cd ..\..
if not exist lpmalgos mkdir lpmalgos
for /r "sources\lpmalgos" %%f in (lpmalgos*.pyd) do copy /y "%%f" lpmalgos\

echo === [4/4] Compilando skeleton (Cython) ===
cd sources\skeleton
if exist build rd /s /q build
python setup.py build_ext --inplace
cd ..\..
if not exist skeleton mkdir skeleton
for /r "sources\skeleton" %%f in (skeleton*.pyd) do copy /y "%%f" skeleton\

echo === [5/5] Testando importacao ===
python -c "import lpmalgos; import skeleton; print('-> Tudo configurado e pronto no Windows!')"
pause