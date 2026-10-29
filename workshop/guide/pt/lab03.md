# Lab 3 — Embeddings: cada token vira um vetor

*Objetivo: dar a cada token uma lista de números que o modelo pode ajustar.*

### Por quê
Um ID é só um rótulo: o ID 8 não é "maior" nem "parecido" com o ID 7. O modelo precisa de números com os quais possa fazer contas e que possam mudar durante o treino.

A solução é uma tabela: cada token recebe uma linha com **{{n_embd}} números**, o seu **embedding** (um vetor). No início, os números são aleatórios. Durante o treino, eles são ajustados para ajudar nas previsões.

Esses números são **parâmetros**: os valores que o modelo aprende. Esta tabela tem {{vocab_size}} × {{n_embd}} = **{{params_embedding}}** parâmetros.

### Preveja
Antes do treino, letras parecidas, como "a" e "e", terão vetores próximos?
@answer **Não.** Os vetores começam aleatórios. Qualquer organização que aparecer depois virá do treino. Volte a este painel depois do Lab 7.

### Copie e cole
@code b03_embeddings
- `nn.Module` é a base de toda peça de rede neural no PyTorch; `super().__init__()` prepara essa base.
- `nn.Embedding(vocab_size, n_embd)` cria a tabela: uma linha por token, `n_embd` números por linha.
- `forward` é o que acontece quando a peça é usada: cada ID vira a sua linha da tabela, e a forma passa de `(B, T)` para `(B, T, n_embd)`.

### Verifique e observe
@verify
No painel:
- o mapa de calor mostra os vetores dos caracteres do texto digitado: uma linha por caractere, {{n_embd}} colunas;
- caracteres repetidos têm linhas idênticas, porque usam a mesma linha da tabela;
- o gráfico de pontos projeta os {{vocab_size}} vetores em 2 dimensões (PCA). É só uma sombra dos {{n_embd}} números, mas ajuda a ver quem está perto de quem.

### Experimente
- Digite uma palavra com letras repetidas, como "arara", e encontre as linhas iguais.
- Observe a projeção: os pontos estão espalhados ao acaso?

### Para pensar
O GPT-2 tem um vocabulário de 50.257 tokens. Com vetores de 128 números, quantos parâmetros teria só a tabela de embeddings?
@answer 50.257 × 128 = **6.432.896** parâmetros. É um dos motivos para este workshop usar caracteres: com {{vocab_size}} tokens e vetores de {{n_embd}} números, a nossa tabela tem {{params_embedding}}.
