# Lab 7 — Treinamento: o modelo aprende

*Objetivo: repetir milhares de vezes o ciclo que ajusta os parâmetros para reduzir a loss.*

### Por quê
Até aqui, os {{params_bigram}} parâmetros do modelo são aleatórios. Treinar é ajustá-los aos poucos para a loss cair. Cada **passo de treino** faz cinco coisas:

1. **Prever:** o modelo calcula os logits de um batch.
2. **Medir o erro:** calcula a loss (Lab 6).
3. **Limpar:** zera os gradientes do passo anterior.
4. **Calcular os gradientes** (`backward`): para cada parâmetro, descobre em que direção mexer para a loss cair. O PyTorch faz essa conta sozinho.
5. **Atualizar:** o otimizador (AdamW) move cada parâmetro um pouco nessa direção.

O tamanho desse "pouco" é a **learning rate** (taxa de aprendizado).

### Preveja
Uma learning rate 10 vezes maior faz o modelo aprender 10 vezes mais rápido?
@answer **Nem sempre.** Passos grandes podem passar do ponto e fazer a loss oscilar; passos pequenos demais deixam o aprendizado lento. Teste no painel.

### Copie e cole
@code b07_training
- `make_optimizer` cria o AdamW, que vai ajustar todos os parâmetros do modelo.
- `train_step` é um passo completo: prever, medir o erro, limpar, calcular os gradientes e atualizar.
- `loss.item()` devolve a loss como um número comum, para desenhar o gráfico.

### Verifique e observe
@verify
Depois da verificação, clique em **Iniciar** no painel e acompanhe:
- as curvas de loss de **treino** e de **validação**, atualizadas a cada 250 passos;
- a linha tracejada: o **limite do bigrama**, {{bigram_floor}}. É a menor loss de treino possível para um modelo que olha só o caractere atual, calculada contando os pares de caracteres do texto de treino. O nosso modelo pode ficar um pouco acima dela;
- a **galeria da evolução**: o mesmo pedido de texto, com a mesma seed, em vários momentos do treino;
- a tabela {{vocab_size}} × {{vocab_size}} ganhando estrutura.

### Experimente
- Treine com learning rate 0,1, 0,01 e 0,001 e compare as curvas. Entre um treino e outro, clique em **Reiniciar modelo**: sem isso, **Iniciar** continua treinando o mesmo modelo. As curvas anteriores ficam no gráfico, em linhas finas.
- Depois do treino, volte aos Labs 3, 4, 5 e 6: os painéis agora mostram o modelo treinado. Os vetores se organizaram? As probabilidades fazem sentido? A loss caiu?

### Para pensar
A loss parou perto do limite do bigrama. O que falta para o modelo escrever melhor?
@answer **Contexto.** Olhando só um caractere, nenhum ajuste de parâmetros passa desse limite. O próximo passo é deixar cada token olhar para os anteriores: a **self-attention**, a partir do Lab 8.
