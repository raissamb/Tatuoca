# Projeto Tatuoca

O campo magnético terrestre, um escudo natural contra a radiação solar, é estudado há séculos por sua importância em áreas como geofísica e ciências espaciais.
O Observatório Magnético de Tatuoca, situado em uma região de interesse geomagnético, tem um papel essencial na coleta de dados sobre as variações temporais
do campo magnético. Seus registros históricos são valiosos para documentar mudanças seculares, diurnas e ocasionais na região equatorial.

Contudo, a preservação desses registros analógicos, principalmente magnetogramas em papel, está ameaçada pela degradação natural e condições ambientais
adversas. A perda desses dados prejudicaria estudos sobre a dinâmica do núcleo terrestre, a interação Sol-Terra, a modelagem do campo magnético e a análise
de riscos geomagnéticos, como tempestades solares.

Para enfrentar esse desafio, o Projeto Tatuoca propõe a digitalização, preservação e disponibilização desses dados para a comunidade científica global.
Utilizando técnicas avançadas de processamento de imagens e análise de dados, o projeto transforma os registros analógicos em informações digitais de alta qualidade.

# Instalação, Compilação e Uso (Windows)

- Clone esse repositório no seu pc através do comando "git clone --recursive-submodules https://github.com/keanusales/Tatuoca.git".
- Baixe e instale a versão mais recente do python pelo site https://www.python.org/downloads/.
- Preste bastante atenção à arquitetura usada na compilação do python, pois ela será útil posteriormente.
- Assegure-se que você também baixou os debug binaries e os dubugging symbols, necessários para compilar extensões ao python.
- Pelo pip, execute o comando "pip install opencv cython easycython" para instalar as dependências necessárias.
- Baixe e instale as ferramentas de compilação do MSVC pelo link https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/.
- Assegure-se de que apenas os componentes individuais a seguir estão instalados:
  - SDK mais recente do Windows, seja Windows 10 ou 11, a depender de qual está instalado no seu pc.
  - Ferramentas do Cmake do C++ para Windows.
  - Ferramentas de compilação MSVC mais recente para sua arquitetura (geralmente a opção anterior já marca essa automaticamente).
  - Atualização do pacote redistribuível do C++.
- Após isso, baixe o Visual Studio Code disponível pelo site https://code.visualstudio.com/download.
- Baixe a extensão compiladora https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools.
- A partir do repositório clonado, abra a pasta /sources/lpmalgos no VSCode. Isso irá carregar o compilador do Cmake.
- Abra a aba de configurações do Cmake no VSCode. Na parte de configurar, altere o kit de compilação para o equivalente à arquitetura do python instalado no seu pc.
- Compile a extensão e a recorte da pasta /sources/lpmalgos/build/bin/release para a pasta /lpmalgos.
- Agora, abra a pasta /sources/skeleton no VSCode e abra um terminal apertando ctrl + shift + '.
- Execute o comando "easycython.exe .\skeleton.pyx" no seu terminal. Isso irá compilar a extensão.
- Recorte a extensão compilada da pasta /sources/skeleton para a pasta /skeleton.
- Pronto! Agora podemos usar o código do Tatuoca!

## Instalação para colaborador

No terminal, clone o projeto com submódulos:

```git clone --recursive [https://github.com/SEU_USUARIO/Tatuoca.git](https://github.com/SEU_USUARIO/Tatuoca.git)```  

Entre na pasta e dê dois cliques no arquivo setup_windows.bat.Ele vai instalar o ambiente virtual e compilar os módulos automaticamente.
