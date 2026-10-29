# Lab 1 — Tokenizador: texto vira números

*Objetivo: transformar texto em uma lista de números (IDs) e voltar ao texto.*

### Por quê
Um modelo de linguagem não lê letras: ele só faz contas com números. Por isso, o primeiro passo é cortar o texto em pedaços, chamados **tokens**, e dar um número (um **ID**) a cada um. O conjunto de todos os tokens conhecidos é o **vocabulário**.

Aqui, cada caractere é um token. Modelos como o GPT usam pedaços maiores, como partes de palavras, mas a ideia é a mesma.

### Preveja
Quantos caracteres diferentes existem em "o rato roeu a roupa do rei de roma"?
@answer **11**: espaço, a, d, e, i, m, o, p, r, t, u.

### Copie e cole
@code b01_tokenizer
- `sorted(set(text))` monta o vocabulário: cada caractere uma única vez, em ordem.
- `stoi` (*string to index*) e `itos` (*index to string*) são os dois dicionários de tradução: caractere → ID e ID → caractere.
- `encode` transforma texto em IDs; `decode` faz o caminho de volta.

### Verifique e observe
@verify
No painel:
- com o texto base "o rato roeu a roupa do rei de roma", o vocabulário tem 11 caracteres, "roma" vira `[8, 6, 5, 1]` e o `decode` devolve "roma";
- com o corpus **{{corpus_name}}**, o vocabulário tem **{{vocab_size}}** caracteres. É este tokenizador que o resto do workshop usa.

### Experimente
- Codifique "Roma", com R maiúsculo. Aparece `KeyError: 'R'`, e isso é esperado: o tokenizador só conhece os caracteres que viu no texto base. Muitos LLMs reais, como o GPT-2, partem de bytes e por isso nunca encontram um caractere desconhecido.
- Troque o texto base por uma frase sua e veja o vocabulário mudar.
- Na tabela do corpus, procure os caracteres mais e menos frequentes.

### Para pensar
A cada passo, entre quantas opções o modelo vai escolher o próximo token?
@answer **{{vocab_size}}**, o tamanho do vocabulário. No Lab 4, esse número vira a última dimensão da saída do modelo.
