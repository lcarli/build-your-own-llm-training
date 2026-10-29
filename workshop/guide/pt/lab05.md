# Lab 5 — Softmax, temperatura e geração de texto

*Objetivo: transformar pontuações em probabilidades e gerar texto, um token por vez.*

### Por quê
Logits são pontuações soltas. A **softmax** as transforma em **probabilidades**: todas positivas e somando 1.

A **temperatura** ajusta a ousadia do modelo: os logits são divididos por ela antes da softmax.
- Temperatura baixa (menor que 1): o favorito fica ainda mais provável, e o texto fica repetitivo.
- Temperatura alta (maior que 1): as chances ficam mais parecidas, e o texto fica mais imprevisível.

Gerar texto é repetir três passos: calcular as probabilidades do próximo token, **sortear** um token e acrescentá-lo ao final. Depois, tudo de novo.

### Preveja
O que acontece com as probabilidades quando a temperatura se aproxima de 0?
@answer Quase toda a probabilidade vai para o token de maior pontuação. No limite, o modelo sempre escolhe o favorito.

### Copie e cole
@code b05_sampling
- `logits[:, -1, :]` pega só a última posição: é ela que prevê o próximo token.
- `F.softmax(..., dim=-1)` transforma cada linha de pontuações em probabilidades.
- `torch.multinomial` sorteia um token, respeitando as probabilidades.
- `torch.cat` acrescenta o token sorteado ao final da sequência.
- `@torch.no_grad()` avisa o PyTorch que não vamos treinar aqui, o que economiza memória e tempo.

### Verifique e observe
@verify
No painel:
- as 10 maiores probabilidades para o próximo caractere e a soma de todas elas: 1;
- um texto de 200 caracteres gerado pelo **seu** `generate`;
- os primeiros passos da geração: o caractere sorteado, a probabilidade dele e as alternativas.

### Experimente
- Mova a temperatura para 0,1, depois para 1 e para 2, e observe as barras.
- Gere duas vezes com a mesma seed: o texto é idêntico. Mude a seed e compare.

### Para pensar
O texto gerado parece uma sopa de letras. Por quê?
@answer Porque o modelo ainda **não treinou**: as pontuações são aleatórias. No Lab 6 medimos o tamanho do erro, e no Lab 7 o modelo aprende.
