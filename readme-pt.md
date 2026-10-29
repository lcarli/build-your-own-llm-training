# Build Your Own LLM — Laboratório Interativo

[English](README.md) | **Português** | [Français](readme-fr.md)

**Convenção deste projeto:** comentários, docstrings e mensagens do código devem ser sempre escritos em inglês. A interface usa inglês; os guias e os textos de treinamento mantêm seus idiomas.

Workshop guiado em que cada estudante monta, peça por peça, um **modelo de linguagem de caracteres** (um bigrama nesta versão; o mini-GPT vem nos próximos laboratórios). Tudo roda **no próprio computador, na CPU**: sem nuvem, sem GPU, sem serviços pagos e sem GitHub Copilot.

Em cada laboratório, uma página local mostra o guia com a explicação e o código pronto. O estudante **copia**, **cola** no arquivo indicado do VS Code e clica em **Verify**. A página então libera uma visualização com os números **reais** do modelo dele: nada de dados inventados.

| Lab | Tema | Arquivo do estudante |
|---|---|---|
| 0 | Preparação: o fluxo do workshop | `bricks/b00_warmup.py` |
| 1 | Tokenizador: texto vira números | `bricks/b01_tokenizer.py` |
| 2 | Exemplos de treino (batches) | `bricks/b02_batches.py` |
| 3 | Embeddings | `bricks/b03_embeddings.py` |
| 4 | Logits: a primeira previsão | `bricks/b04_bigram.py` |
| 5 | Softmax, temperatura e geração | `bricks/b05_sampling.py` |
| 6 | Loss: medindo o erro | `bricks/b06_loss.py` |
| 7 | Treinamento | `bricks/b07_training.py` |

## Requisitos

- **Python 3.10 ou mais novo** (Windows, macOS ou Linux).
- **PyTorch** (versão CPU). É a única dependência: o servidor usa só a biblioteca padrão do Python e a página é HTML/CSS/JS puro, sem bibliotecas externas.
- Um navegador atual (Chrome, Edge, Firefox ou Safari) e o **VS Code**.
- Qualquer notebook comum: o modelo tem poucos milhares de parâmetros e treina em segundos.

## Preparação (instrutor)

1. **Crie um ambiente virtual** na pasta do projeto:

   | Sistema | Comandos |
   |---|---|
   | Windows (PowerShell) | `py -3 -m venv .venv` e `.venv\Scripts\Activate.ps1` |
   | Windows (cmd) | `py -3 -m venv .venv` e `.venv\Scripts\activate.bat` |
   | macOS / Linux | `python3 -m venv .venv` e `source .venv/bin/activate` |

   Se o PowerShell bloquear o script de ativação: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

2. **Instale o PyTorch para CPU** (download bem menor que a versão com GPU):

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Prepare o texto de treino** (`data/corpus.txt`) no idioma da turma:

   ```
   python data/prepare_data.py              # português: Dom Casmurro (padrão)
   python data/prepare_data.py --lang fr    # francês: Maria Chapdelaine
   python data/prepare_data.py --lang en    # inglês: Anne of Green Gables
   python data/prepare_data.py --list       # todos os textos disponíveis
   ```

   | id | Idioma | Livro | Por que este |
   |---|---|---|---|
   | `machado` (padrão pt) | português | *Dom Casmurro*, Machado de Assis (1899) | Clássico brasileiro. Usa a ortografia da edição de 1899 ("elle", "cançado"). |
   | `hemon` (padrão fr) | francês | *Maria Chapdelaine*, Louis Hémon (1913) | Clássico da literatura quebequense, passado no interior do Québec. |
   | `verne` | francês | *Vingt mille lieues sous les mers*, Jules Verne (1869-1870) | Aventura clássica; texto longo, bom para juntar com outro livro. |
   | `montgomery` (padrão en) | inglês | *Anne of Green Gables*, L. M. Montgomery (1908) | Clássico canadense, de leitura fácil. |
   | `shakespeare` | inglês | *Tiny Shakespeare* (seleção de Andrej Karpathy) | O texto do nanoGPT: permite comparar com os resultados publicados. |

   Outras opções:
   - `--corpus hemon verne` junta dois ou mais textos num corpus só (útil para ter mais texto em francês);
   - `--file meu_texto.txt` ou `--url https://...` usa um texto seu;
   - `--keep-lines` mantém as quebras de linha originais (por padrão, os romances ficam com um parágrafo por linha).

   O script remove o cabeçalho e a licença do Project Gutenberg, troca os espaços tipográficos (comuns antes de `; : ! ?` em francês) por espaços normais e apaga caracteres invisíveis. Rode-o uma vez, na sua máquina, e distribua a pasta com o corpus pronto: assim a turma não depende da internet. Sem `data/corpus.txt`, o workshop usa um texto curto de demonstração (`data/sample_pt.txt`) e mostra um aviso.

4. **Rode o autoteste** (verifica os 8 laboratórios com o código de referência, todos os painéis e um treino de 300 passos):

   ```
   python tools/selftest.py
   ```

   Ele não mexe em `bricks/` nem em `.workshop/`.

### Pacote offline para a sala

Se a rede da sala for ruim, leve as rodas (wheels) do PyTorch num pendrive. Em cada combinação de sistema e versão do Python que os estudantes usam:

```
pip download torch --index-url https://download.pytorch.org/whl/cpu -d wheels/
```

Na sala, com o ambiente virtual ativado:

```
pip install --no-index --find-links wheels torch
```

A roda precisa combinar com o sistema (Windows, macOS, Linux), a arquitetura (x86-64 ou ARM) e a versão do Python (3.10, 3.11, 3.12...). **Teste o pacote uma vez numa máquina limpa** antes do workshop. Distribua a pasta do projeto já com `data/corpus.txt` preparado.

## Como rodar

```
python run.py                  # modo Build: usa o código dos estudantes em bricks/
python run.py --explore        # modo Explore: usa o código de referência e libera todos os labs
python run.py --port 8001      # outra porta
python run.py --no-browser     # não abre o navegador sozinho
python run.py --lang pt        # Portuguese guides (default)
python run.py --lang en        # English guides
python run.py --lang fr        # French guides
```

Os 8 guias estão em [`workshop/guide/pt/`](workshop/guide/pt/), [`workshop/guide/en/`](workshop/guide/en/) e [`workshop/guide/fr/`](workshop/guide/fr/). `--lang` funciona também com `--explore` e muda apenas o idioma dos guias, sem alterar o código, a interface em inglês, o corpus ou o progresso. Para mudar o texto de treinamento, use separadamente `python data/prepare_data.py --lang pt|en|fr`.

O endereço é `http://127.0.0.1:8000/`. O servidor só aceita conexões do próprio computador, a página não carrega nada da internet e nenhum dado sai da máquina. Para encerrar: **Ctrl+C** no terminal.

O progresso fica em `.workshop/` (`progress.json`, `model.pt`): ao rodar de novo, os laboratórios concluídos são verificados outra vez em silêncio e o modelo treinado é recarregado. Para recomeçar do zero, encerre o servidor e apague a pasta `.workshop/`.

O **modo Explore** serve para o instrutor demonstrar os painéis ou para quem quer só explorar: ele não usa nem altera os arquivos de `bricks/` e guarda o estado em `.workshop/explore/`.

## Fluxo do estudante

1. No VS Code, abra **a pasta do projeto** (Arquivo → Abrir Pasta). O salvamento automático já vem ligado em `.vscode/settings.json`.
2. Rode `python run.py` e use a página que abre no navegador.
3. Em cada laboratório: leia o guia → clique em **Copy** → abra o arquivo indicado (`bricks/bNN_....py`) → cole **abaixo** da linha `# ---- PASTE BELOW ----` → volte à página e clique em **Verify**.
4. A verificação mostra cada teste com ✓ ou ✗. Quando tudo passa, a visualização da direita é liberada e o próximo laboratório também.
5. Quem quiser escrever o código sozinho pode clicar em **I want to write it** para esconder o código do guia.

## Solução de problemas

### Mensagens de erro ao verificar

As mensagens em inglês são traduzidas abaixo para referência.

| Mensagem | Causa comum | O que fazer |
|---|---|---|
| Ainda não há código abaixo da linha `# ---- PASTE BELOW ----` | O arquivo não foi salvo, ou o código foi colado em outro arquivo | Aperte Ctrl+S no VS Code e confira o nome do arquivo |
| Aspas curvas no código | Código copiado de slides, Word, PDF ou chat | Copie pelo botão **Copy** do guia |
| Caractere invisível no código (U+00A0, U+200B...) | Código passou por e-mail, documento ou chat | Apague os espaços da linha e digite de novo, ou use **Copy** |
| Problema de recuo / Recuo inesperado | A seleção perdeu (ou ganhou) espaços no começo das linhas | Use **Copy**; no VS Code, Tab e Shift+Tab ajustam o recuo |
| O final do código ficou de fora | Seleção incompleta | Use **Copy**, que pega o bloco inteiro |
| Este código parece ser do Lab N | Código colado no arquivo errado | Recorte e cole no arquivo indicado na mensagem |
| O PyTorch não está instalado neste Python | O ambiente virtual não foi ativado, ou o PyTorch não foi instalado | Ative o `.venv` e instale o PyTorch (veja abaixo) |

Erros dentro do código (por exemplo, `TypeError` ou formas de tensores que não combinam) aparecem com o tipo do erro, a mensagem e a linha do arquivo do estudante.

### Completar este passo (plano B)

O link **Complete this step**, abaixo do botão **Verify**, troca o conteúdo do arquivo do laboratório pelo código de referência e verifica em seguida. Uma cópia do arquivo anterior fica em `.workshop/backup/`. Use quando alguém travar e o grupo precisar seguir.

### A porta já está em uso

`Port 8000 is already in use`: o workshop provavelmente já está aberto em outro terminal. Feche-o ou rode `python run.py --port 8001`.

### Instalação do PyTorch por sistema

- **Windows:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Se aparecer um erro de DLL (por exemplo, `WinError 126`), instale o *Microsoft Visual C++ Redistributable* (x64) e reinicie o computador.
- **macOS com Apple Silicon (M1 ou mais novo):** `pip install torch`.
- **macOS com processador Intel:** o PyTorch oficial parou de publicar versões para Mac Intel depois da 2.2, com instaladores só para Python 3.10 a 3.12. Use Python 3.12 ou anterior: `pip install torch` instala a 2.2.2, que tem tudo o que o workshop usa. Rode o autoteste para confirmar.
- **Linux:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` evita baixar a versão com CUDA, que é muito maior.

Confira sempre se o `pip` é o do ambiente virtual: `python -m pip install ...` garante isso.

### A verificação demora

A primeira verificação do Lab 0 mede a velocidade do computador (cerca de meio segundo). Se a verificação passar de 20 segundos, o código provavelmente tem um laço infinito: encerre o servidor (Ctrl+C) e rode `python run.py` de novo.

## Estrutura do projeto

```
build-your-own-llm/
├── run.py                  # ponto de entrada: python run.py [--port 8000] [--explore] [--no-browser]
├── requirements.txt        # só o torch
├── bricks/                 # arquivos dos estudantes (b00 ... b07), com a linha marcadora
├── reference/              # código de referência de cada laboratório (o mesmo do guia)
├── data/
│   ├── prepare_data.py     # baixa ou prepara data/corpus.txt
│   └── sample_pt.txt       # texto curto de demonstração
├── workshop/               # motor do workshop (servidor, verificações, treino, guias)
│   └── guide/              # guias: pt/, en/ e fr/, cada um com lab00.md ... lab07.md
├── web/                    # página: index.html, style.css, app.js, charts.js, md.js
├── tools/selftest.py       # autoteste para o instrutor
└── .workshop/              # criado ao rodar: progresso, modelo treinado, backups
```

Os testes dos guias e da seleção de idioma podem ser executados com `python -B -m unittest tools.test_guides`. Não exigem PyTorch; a verificação do renderizador usa Node.js.

## Próximos laboratórios

| Lab | Tema |
|---|---|
| 8 | Posição e máscara causal |
| 9 | Self-attention |
| 10 | Multi-head + MLP |
| 11 | Bloco Transformer |
| 12 | Mini-GPT |
| 13 | Playground |

## Créditos

- A estrutura do workshop se inspira nos projetos **nanoGPT** e **makemore**, de Andrej Karpathy (licença MIT).
- **Tiny Shakespeare**: texto reunido por Andrej Karpathy no projeto *char-rnn*, a partir das obras de William Shakespeare (domínio público).
- Textos do **Project Gutenberg**, em domínio público (autores falecidos há mais de 80 anos): *Dom Casmurro*, de Machado de Assis (eBook #55752); *Maria Chapdelaine*, de Louis Hémon (#13525); *Vingt mille lieues sous les mers*, de Jules Verne (#5097); *Anne of Green Gables*, de L. M. Montgomery (#45).
- `data/sample_pt.txt` é um texto original escrito para este workshop.
