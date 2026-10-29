# Lab 2 — Exemplos de treino: um texto, milhares de perguntas

*Objetivo: transformar o corpus em pares "contexto → próximo token" e separar treino e validação.*

### Por quê
O modelo aprende uma única tarefa: **prever o próximo token**. O próprio texto já traz as respostas certas. Numa janela de 8 caracteres, o modelo treina 8 previsões: com o 1º caractere, prever o 2º; com os 2 primeiros, prever o 3º; e assim por diante, até prever o caractere que vem logo depois da janela.

O texto também é dividido em duas partes:
- **treino** (90%): o modelo aprende com ele;
- **validação** (10%): o modelo nunca treina com ele; serve para medir se aprendeu algo que vale para texto novo.

Para treinar, sorteamos vários trechos de uma vez. Esse grupo de trechos é um **batch**.

### Preveja
Uma janela `x` tem 8 tokens. Quantos exemplos de treino (contexto → alvo) ela contém?
@answer **8**. O alvo `y` é a mesma janela deslocada uma posição para a frente: cada posição de `x` tem a sua resposta na mesma posição de `y`.

### Copie e cole
@code b02_batches
- `torch.tensor` transforma a lista de IDs num **tensor**, a estrutura de números do PyTorch.
- `split_train_val` corta o tensor em duas partes: 90% e 10%.
- `get_batch` sorteia `batch_size` posições de início e recorta duas janelas a partir de cada uma: `x`, o contexto, e `y`, a mesma janela uma posição à frente.
- `torch.stack` empilha as janelas numa tabela de forma `(B, T)`: B trechos com T tokens cada.

### Verifique e observe
@verify
No painel:
- o treino tem **{{train_tokens}}** tokens e a validação, **{{val_tokens}}**;
- um batch com {{batch_size_demo}} trechos de {{block_size_demo}} tokens: compare `x` e `y` e encontre o deslocamento de uma posição;
- os {{block_size_demo}} pares "contexto → alvo" da primeira linha.

### Experimente
- Clique em **Novo batch** para sortear outros trechos.
- Mude o tamanho da janela (*block size*) e conte os pares de novo.

### Para pensar
Por que guardar 10% do texto que o modelo nunca vê durante o treino?
@answer Porque um modelo pode **decorar** o texto de treino. A loss na validação mostra se ele aprendeu padrões que funcionam em texto novo. As duas curvas aparecem lado a lado no Lab 7.
