#!/usr/bin/env bash

# Encerra o script caso qualquer comando falhe
set -e

# --- CONFIGURAÇÃO ---
# Substitua pela URL do seu fork se necessário
REPO_URL="https://github.com/raissamb/Tatuoca.git"
TARGET_DIR="Tatuoca"
BRANCH_NAME="linuxdev"

echo "=== [1/6] Instalando dependências essenciais do sistema (requer sudo) ==="
sudo apt update
sudo apt install -y git build-essential cmake python3-dev python3-venv libomp-dev

echo "=== [2/6] Clonando repositório de forma recursiva ==="
if [ ! -d "$TARGET_DIR" ]; then
    git clone --recursive "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
    git checkout "$BRANCH_NAME" || echo "Aviso: branch $BRANCH_NAME não encontrada, mantendo a padrão."
else
    echo "Diretório $TARGET_DIR já existe. Atualizando submódulos..."
    cd "$TARGET_DIR"
    git submodule update --init --recursive
fi

echo "=== [3/6] Criando e configurando ambiente virtual Python ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install setuptools cython numpy opencv-python pandas matplotlib scipy

echo "=== [4/6] Compilando extensão C++ (lpmalgos) ==="
cd sources/lpmalgos
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Release -B build/
cmake --build build/ --config Release

cd ../..
mkdir -p lpmalgos/
find sources/lpmalgos/ -name "lpmalgos*.so" -exec cp {} lpmalgos/ \;

echo "=== [5/6] Compilando extensão Cython (skeleton) ==="
cd sources/skeleton
rm -rf build/ skeleton.cpp skeleton.html
python3 setup.py build_ext --inplace

cd ../..
mkdir -p skeleton/
cp sources/skeleton/skeleton.*.so skeleton/

echo "=== [6/6] Verificando instalação ==="
python3 -c "import lpmalgos; import skeleton; print('-> Tudo instalado e importado com sucesso!')"

echo "=== Concluído! Para começar a usar, acesse a pasta e ative o ambiente: ==="
echo "cd $TARGET_DIR && source .venv/bin/activate"