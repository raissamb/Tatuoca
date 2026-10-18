# Guia de Configuração e Compilação no Linux (Projeto Tatuoca)

Este guia documenta o fluxo de configuração do ambiente e compilação nativa das extensões C++ (`lpmalgos`) e Cython (`skeleton`) em distribuições Linux (Ubuntu/Debian).

---

## 1. Pré-requisitos do Sistema Operacional

Instale os compiladores, cabeçalhos de desenvolvimento do Python e utilitários de build via terminal:

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-dev python3-venv libomp-dev


### Script de Automação Completa (`bootstrap_linux.sh`)

Este script pode ser salvo na sua pasta pessoal (por exemplo, em `~/Documentos/envs/bootstrap_linux.sh`). Ele automatiza a instalação das ferramentas do sistema com `sudo`, clona o seu repositório com submódulos, entra na branch `linuxdev` e roda a compilação completa.

Crie o arquivo:
```bash
nano bootstrap_linux.sh
