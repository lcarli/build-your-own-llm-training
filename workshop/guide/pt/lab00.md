# Lab 0 — Preparação: o fluxo do workshop

*Objetivo: conferir que o seu computador está pronto e treinar o ciclo de todos os laboratórios: ler, copiar, colar e verificar.*

### Por quê
Neste workshop você vai montar um modelo de linguagem peça por peça, como LEGO. Todo laboratório segue o mesmo ciclo:

1. Leia a explicação.
2. Clique em **Copy** no bloco de código.
3. No VS Code, abra o arquivo indicado e cole **abaixo** da linha `# ---- PASTE BELOW ----`. O arquivo é salvo automaticamente.
4. Volte para esta página e clique em **Verify**.
5. A visualização à direita é liberada, sempre com os números reais do **seu** modelo.

Tudo roda no seu computador: sem nuvem, sem GPU e sem conta paga.

### Preveja
Quantas operações com números decimais (multiplicações e somas) o seu computador faz por segundo? Mil? Um milhão? Um bilhão?
@answer A verificação mede isso no seu computador, multiplicando tabelas de números. O resultado aparece em **GFLOP/s**: bilhões de operações por segundo. Compare com a pessoa ao lado.

### Copie e cole
Este primeiro código é só um aquecimento: uma função que devolve uma saudação.
@code b00_warmup
- `def` cria uma função; `name` é o valor que ela recebe.
- O texto entre aspas triplas descreve o que a função faz.
- `f"..."` monta um texto com o valor de `name` dentro.

### Verifique e observe
@verify
A verificação confere:
- a versão do Python (3.10 ou mais nova);
- se o PyTorch está instalado e com quantos núcleos do processador ele vai trabalhar;
- o texto de treino: **{{corpus_name}}**, com {{corpus_chars}} caracteres;
- se a pasta `bricks/` aceita gravação;
- a sua função `greet`;
- a velocidade do seu computador para multiplicar matrizes.

### Experimente
Erre de propósito para conhecer as mensagens de ajuda:
1. Em `bricks/b00_warmup.py`, apague os espaços do início da segunda linha do código (a que começa com `"""`) e clique em **Verify**. A mensagem aponta a linha e explica o problema de recuo.
2. Desfaça (Ctrl+Z). Agora troque as aspas retas `"` da linha do `return` por aspas curvas `“` e `”`, como as que o Word e o PowerPoint inserem. Verifique de novo.
3. Corrija o arquivo, ou clique em **Complete this step** para restaurar o código de referência.

> Copie o código sempre pelo botão **Copy** deste roteiro. Slides, PDFs, e-mails e chats trocam aspas e espaços por caracteres que o Python não aceita.

### Para pensar
Por que este workshop roda tudo no seu computador, em vez de usar um serviço na nuvem?
@answer Para que cada número da tela venha do **seu** modelo, sem custo e sem depender de internet. E para mostrar que o mecanismo de um LLM cabe num notebook: nos modelos grandes, o que muda é principalmente a escala (mais dados, mais parâmetros e muito mais computação).
