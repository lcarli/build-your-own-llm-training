# Lab 4 — Logits: a primeira previsão

*Objetivo: montar o primeiro modelo completo, que dá uma pontuação a cada possível próximo token.*

### Por quê
Cada token já tem um vetor. Agora precisamos transformar esse vetor em **uma pontuação para cada um dos {{vocab_size}} tokens** que podem vir a seguir. Essas pontuações se chamam **logits**: quanto maior, mais o modelo aposta naquele token.

Quem faz isso é uma camada linear (`nn.Linear`): entram {{n_embd}} números, saem {{vocab_size}} pontuações.

Juntando o embedding (Lab 3) e a camada linear, temos um modelo **bigrama**: ele prevê o próximo token olhando **só para o token atual**.

### Preveja
Um logit pode ser negativo?
@answer **Sim.** Logit é uma pontuação, não uma probabilidade. No Lab 5, a softmax transforma as pontuações em probabilidades.

### Copie e cole
@code b04_bigram
- `from bricks.b03_embeddings import TokenEmbedding` reaproveita a peça do Lab 3: é o LEGO em ação.
- `self.lm_head` é a camada linear que transforma cada vetor em {{vocab_size}} pontuações.
- No `forward`, os IDs viram vetores e os vetores viram logits: `(B, T)` → `(B, T, {{n_embd}})` → `(B, T, {{vocab_size}})`.

### Verifique e observe
@verify
No painel:
- o caminho das formas: `(1, T)` → `(1, T, {{n_embd}})` → `(1, T, {{vocab_size}})`;
- as pontuações para o caractere que viria depois do último caractere digitado;
- a tabela {{vocab_size}} × {{vocab_size}}: cada linha é um caractere atual, cada coluna é um próximo caractere possível. Sem treino, o padrão é aleatório.

O modelo inteiro tem **{{params_bigram}}** parâmetros.

### Experimente
- Mude o último caractere do texto e veja as pontuações mudarem.
- Encontre na tabela a linha do caractere que você digitou.

### Para pensar
O que este modelo **não** consegue saber sobre o texto?
@answer Tudo o que veio **antes** do caractere atual. Depois de "qu" ou de um "u" sozinho, ele dá exatamente as mesmas pontuações. Resolver isso é o papel da atenção, a partir do Lab 8.
