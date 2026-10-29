# Lab 6 — Loss: medindo o erro

*Objetivo: calcular um único número que diz o quanto o modelo erra.*

### Por quê
Para aprender, o modelo precisa saber o quanto errou. A medida usada pelos LLMs é a **cross-entropy** (entropia cruzada):

1. Em cada posição, veja a probabilidade que o modelo deu ao token **real** que veio depois.
2. Calcule −log dessa probabilidade: é a "surpresa" do modelo.
3. Tire a média de todas as posições.

Se o modelo deu probabilidade 1 ao token certo, a surpresa é 0. Se deu uma probabilidade quase nula, a surpresa é enorme. Essa média é a **loss**: quanto menor, melhor.

### Preveja
Qual é a loss de um modelo que dá a mesma probabilidade a todos os {{vocab_size}} tokens?
@answer **ln({{vocab_size}}) ≈ {{ln_vocab}}.** Cada token recebe 1/{{vocab_size}}, e −log(1/{{vocab_size}}) = ln({{vocab_size}}). É a loss de quem "não sabe nada".

### Copie e cole
@code b06_loss
- `logits.shape` devolve as três dimensões: B trechos, T posições e V tokens possíveis.
- `.view(B * T, V)` junta todas as posições numa única lista de previsões; `targets.view(B * T)` faz o mesmo com as respostas certas.
- `F.cross_entropy` aplica a softmax, calcula −log da probabilidade do token certo e tira a média, tudo de uma vez.

### Verifique e observe
@verify
No painel:
- a surpresa do modelo em cada posição do texto digitado (cores mais fortes indicam mais surpresa) e a probabilidade que ele deu ao caractere real;
- a loss média, comparada com {{ln_vocab}}, a loss de um modelo que divide a probabilidade igualmente;
- sem treino, a loss costuma ficar um pouco acima de {{ln_vocab}}: as pontuações aleatórias favorecem alguns tokens sem motivo.

### Experimente
- Compare um texto comum com uma sequência aleatória de letras. Antes do treino, as duas têm loss parecida.
- Volte aqui depois do Lab 7 e repita a comparação.

### Para pensar
Se o modelo acertar sempre, com probabilidade 1, qual é a loss?
@answer **0**, porque −log(1) = 0. Na prática, a loss nunca chega a 0 com texto real: o próximo caractere nunca é totalmente previsível.
