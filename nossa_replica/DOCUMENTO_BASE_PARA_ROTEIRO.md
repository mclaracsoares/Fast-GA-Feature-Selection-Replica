# Documento-base para montar o roteiro de fala e a apresentação

Projeto da disciplina IA Avançada (UNICAP): replicação e melhoria do artigo **"Fast Genetic Algorithm for feature selection — A qualitative approximation approach"**.

> **Instrução para quem for usar este documento (Claude/app):** use-o como fonte única. Todos os números abaixo vêm de execuções reais (10 sementes por algoritmo e configuração, 360 execuções no total), medidos no conjunto de **teste**. Os trechos marcados **[Observação honesta]** são limitações ou incertezas que devem aparecer na apresentação, não ser escondidas. Os nomes de arquivos e figuras podem ser citados nos slides.

---

## PARTE 1 — O artigo original

### 1.1 Referência
- Altarabichi, Nowaczyk, Pashami, Mashhadi. *Expert Systems with Applications* 211 (2023), 118528. arXiv:2404.03996.
- Código oficial: github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection (arquivos `Dataset.py`, `Evolution.py`, `Example.ipynb`).

### 1.2 O problema: seleção de atributos (*feature selection*)
- Uma base de dados tem muitas colunas (atributos). Nem todas ajudam a prever; algumas são ruído ou redundantes.
- **Seleção de atributos** é escolher o subconjunto de colunas que dá o melhor desempenho ao classificador.
- Com 41 atributos existem 2^41 (cerca de 2 trilhões) subconjuntos possíveis. Testar todos é impossível.
- Abordagem **wrapper**: cada subconjunto candidato é avaliado treinando de fato o classificador e medindo seu desempenho. É a mais precisa, mas é cara, porque cada avaliação treina um modelo com cerca de 120 mil instâncias.

### 1.3 Como o artigo resolve: meta-heurísticas
Em vez de testar tudo, uma meta-heurística "evolui" uma população de subconjuntos. Cada subconjunto é um vetor binário de 41 posições (1 = atributo escolhido, 0 = descartado). O artigo usa dois algoritmos:

1. **CHC** (um algoritmo genético; descrição conferida em `Evolution.py`):
   - População de 50 indivíduos.
   - Os indivíduos são pareados ao acaso. Um par só se cruza se a **distância de Hamming** entre eles for maior que um limiar `d` (inicialmente `d = tamanho/4`, ou seja, 41//4 = 10). Isso preserva a diversidade.
   - Cruzamento **HUX**: nos bits em que os dois pais diferem, cada filho recebe a metade deles (`len(diferentes)//2`) vinda do outro pai.
   - Filhos repetidos (já vistos antes) são descartados.
   - Seleção elitista: a nova população são os 50 melhores entre pais e filhos.
   - Quando a população não muda numa geração, `d` diminui em 1. Quando `d` chega a 0, ocorre um **reinício**: `d` volta ao valor inicial e a nova população é formada por mutantes do melhor indivíduo (mutação *flip-bit* com probabilidade 1/3 por bit). Ou seja, não há mutação no ciclo normal, só no reinício, e o reinício parte do melhor indivíduo (não é um sorteio totalmente novo).
   - O CHC puro do notebook oficial para quando passa 10 gerações sem melhora (`max_no_change=10`).
2. **PSO** (*Particle Swarm Optimization*, enxame de partículas), via biblioteca `pyswarms`:
   - 50 partículas se movem num espaço contínuo; uma posição acima de 0,5 vira bit 1.
   - Cada partícula é puxada pela sua melhor posição passada (`c1 = 1,49618`) e pela melhor do enxame (`c2 = 1,49618`), com inércia `w = 0,7298`.
   - Topologia Star (todas veem a melhor global).
   - Para quando o resultado não melhora por 10 passos.

### 1.4 O problema: a avaliação é lenta
Toda a busca gasta seu tempo na **fitness** (a nota de cada subconjunto), que exige treinar a árvore de decisão com todas as instâncias de treino.

### 1.5 A contribuição do artigo: aproximação qualitativa (QX)
**Ideia central:** para a busca, o que importa não é o valor exato da acurácia, e sim o **ranking** — saber qual subconjunto é melhor que qual. Um modelo treinado com **menos instâncias** é bem mais barato e, se preservar o ranking, serve igualmente para guiar a busca.

Passos do QX (função `select_instances` e `evaluate_selected_instances` em `Evolution.py`):
1. Sorteia-se `q` subconjuntos de atributos aleatórios e avalia-se cada um com a função **original** (treino completo). Isso dá um ranking de referência.
2. Procura-se uma **amostra de instâncias** (ela mesma encontrada por um algoritmo genético) tal que, treinando só nela, o ranking dos mesmos `q` subconjuntos seja parecido com o de referência.
3. A semelhança é medida pela **correlação de postos de Spearman (ρ)**. Além disso, a amostra é penalizada pelo tempo que gasta. A função de distância é: `distância = (1 − ρ) + tempo_aproximado / tempo_original`. Quanto menor, melhor.
4. A amostra escolhida define o **meta-modelo**: um treino barato que passa a servir de fitness na busca.
5. A cada `f = 10` gerações, a população é reavaliada com a função original ("controle de evolução") para garantir que o meta-modelo não se desviou. O processo termina quando não há melhora por `f_no_change = 2` ciclos.

**Resultado do artigo:** acurácia igual ou melhor, **menor variância** e **muito menos tempo** que CHC e PSO puros.

### 1.6 Base de dados
- **census-income** (UCI Census-Income KDD): 199.523 pessoas, 41 atributos já numéricos/codificados, sem valores faltantes.
- Tarefa: prever se a **renda anual é maior que US$ 50 mil**.
- Divisão **60% treino / 20% validação / 20% teste**, **sequencial, sem embaralhar** (`shuffle=False`), com normalização *z-score* ajustada só no treino.
- A fitness é sempre calculada na **validação**. O teste só é usado para reportar o resultado final.
- Classificador do *wrapper*: `DecisionTreeClassifier(random_state=0)`.

### 1.7 Resultados do artigo (Tabela 2, census-income, acurácia de teste, mediana ± dp, 10 execuções)

| Método | Artigo |
|---|---|
| Baseline (árvore, 41 atributos) | 92,87 |
| CHC | 94,63 ± 0,35 |
| CHC_QX | 94,94 ± 0,07 |
| PSO | 94,79 ± 0,24 |
| PSO_QX | 94,94 ± 0,06 |

Mensagem do artigo: selecionar atributos supera usar todos; os métodos QX igualam ou superam os puros, com menor variância e muito menos tempo.

### 1.8 O que o próprio artigo admite como limitação
Na seção 7 (*Limitations*), os autores escrevem: *"We have never considered the class imbalance case as it is not the focus of this paper. In all experiments accuracy was used to evaluate the fitness of feature subsets. This metric could naturally be replaced with one that accounts for the imbalance case, e.g., average recall."* Foi exatamente aí que encaixamos a nossa melhoria.

---

## PARTE 2 — Nossa replicação: o que fizemos e por quê

### 2.1 Regras que seguimos
- Usar o **código oficial dos autores** sem mudar a lógica de `Evolution.py` e `Dataset.py`.
- Só três correções de compatibilidade (o código é de 2022 e quebra com bibliotecas novas), todas documentadas:
  1. `np.NaN` → `np.nan` em `Dataset.py` (removido no numpy 2).
  2. `delim_whitespace=True` → `sep=r'\s+'` em `Dataset.py` (descontinuado no pandas; esse ramo nem é usado pelo census-income).
  3. Remover a linha `np.warnings.filterwarnings(...)` do `Example.ipynb` (`np.warnings` não existe mais; a linha só silenciava um aviso).
- **Hiperparâmetros exatos do notebook oficial:**
  - `DecisionTreeClassifier(random_state=0)`.
  - `shuffle=False`.
  - População 50.
  - `CHCqx(dataset, 10, 10, 2, 50)`, ou seja, f=10, q=10, f_no_change=2, população 50.
  - `PSOqx(dataset, options, 10, 10, 2, 50)` com `options = {c1: 1.49618, c2: 1.49618, w: 0.7298, k: 50, p: 2}`.
  - CHC puro com `d = ind_size // 4` e `max_no_change = 10`.
  - PSO puro com `steps_no_change = 10`.
- **10 repetições** por algoritmo, com sementes `random.seed(s)` e `np.random.seed(s)` para s = 0..9.
- **Nenhum *grid search*** e nenhum uso do teste para escolher configuração: os valores vêm do artigo.
- `OMP_NUM_THREADS=1`: cada processo usa 1 thread. O paralelismo vem de rodar execuções independentes em processos separados (`joblib`/loky), o que também mantém a medição de tempo comparável.
- Cada execução é salva em disco (`results/runs/<config>__<algo>__s<semente>.json`) assim que termina, de forma atômica. Se o computador desligar, retomamos do ponto em que parou.

### 2.2 Divergências entre o artigo e o código (documentadas; mantivemos o que o código faz)

| Item | Artigo | Código oficial | Impacto |
|---|---|---|---|
| Tamanho `q` da amostra de subconjuntos | q = 20 | q = 10 | Meta-modelo calibrado com menos pontos; correlação de Spearman mais ruidosa, execução mais rápida |
| Orçamento de tempo do CHC | CHC roda o mesmo tempo de CPU que o CHC_QX | CHC para por convergência, sem orçamento de tempo | CHC puro leva bem mais tempo que o QX; a comparação deixa de ser "a tempo igual" |
| Embaralhamento | Instâncias embaralhadas | `shuffle=False` | Divisão determinística; a proporção de positivos é parecida nas três partes, então não distorce |

### 2.3 Estrutura de pastas
```
Fast-Genetic-Algorithm-For-Feature-Selection-main/
├── projeto_original/        ← o projeto dos AUTORES (só 3 correções de compatibilidade)
│   ├── code/                Dataset.py, Evolution.py, Example.ipynb
│   ├── data/census-income.data
│   └── results/results.csv
└── nossa_replica/           ← o NOSSO trabalho
    ├── projeto_replica_e_melhoria.ipynb   (notebook único com tudo)
    ├── melhoria/DatasetMod.py             (a única mudança de lógica da melhoria)
    ├── src/runner.py                      (executa as buscas)
    ├── src/analise.py                     (tabelas, testes estatísticos, gráficos)
    ├── results/                           (runs/*.json e CSVs finais)
    ├── figures/                           (PNGs 150 dpi para os slides)
    └── README.md
```

### 2.4 Arquivos de apoio do nosso projeto
- **`src/runner.py`** — o "motor". Define `CONFIGS` (as 9 configurações), as constantes do artigo e a função `run_one(config, algo, seed)`, que: fixa as sementes; cria o `Dataset` (original ou `DatasetMod`); roda o algoritmo; treina um clone final na base de treino com a solução encontrada; calcula as métricas na validação e no teste (acurácia, acurácia balanceada, precisão, recall, F1, MCC, ROC-AUC e a matriz de confusão); e salva tudo em JSON. `run_many` roda uma lista de execuções em paralelo, ordenadas das mais caras para as mais baratas (balanceamento de carga) e só roda o que ainda não está salvo. Também contém a avaliação de transferência.
- **`src/analise.py`** — tabelas "mediana ± dp", paleta de cores (Okabe-Ito, segura para daltonismo), `salvar_fig`, e `testes_pareados` (Wilcoxon e *t* pareado por semente).
- **`melhoria/DatasetMod.py`** — descrito na Parte 3.

### 2.5 O notebook célula por célula (`projeto_replica_e_melhoria.ipynb`, 52 células)

**Estrutura:** abertura + Parte A (replicação) + Parte B (melhoria) + Parte C (resumo). Cada seção tem um bloco "🎯 O que esta seção faz / 💡 Por que existe". A numeração das seções da Parte A pula A2 e A10 (sobras da versão anterior do notebook); a sequência real é A1, A3, A4, A5, A6, A7, A8, A9, A11. Em várias seções, a célula markdown de "Leitura" fica junto do cabeçalho da seção seguinte; os índices abaixo foram conferidos no arquivo `.ipynb`.

**Célula 0 (markdown) — Abertura.** Título, referências do artigo e do código, tabela das três partes, mapa de pastas, mapa das seções. Por que existe: situar quem lê.

**Célula 1 — Cabeçalho da Parte A.** Objetivo: replicar exatamente o experimento com o código dos autores.

**A1 (célula 2, markdown) — O problema e a ideia do artigo.** Explica seleção de atributos por *wrapper*, algoritmos evolutivos e a aproximação qualitativa (QX). Por que existe: sem entender o que o artigo propõe, não dá para avaliar se a réplica é fiel nem onde a melhoria se encaixa.

**A3 (célula 3, markdown) — Instalação, sementes, correções.** Lista dependências, as 3 correções de compatibilidade (tabela com arquivo, antes, depois e motivo), a política de sementes e o `OMP_NUM_THREADS=1`. Por que existe: transparência sobre o que foi mexido e garantia de reprodutibilidade.

**Célula 4 (código) — Configuração do ambiente.** Força 1 thread por processo (variáveis `OMP/OPENBLAS/MKL_NUM_THREADS`), desliga avisos, localiza as pastas (`RAIZ`, `ORIGINAL`), ajusta o `sys.path` para importar `Dataset`, `runner`, `analise` e `DatasetMod`, importa bibliotecas e define os parâmetros `N_RUNS` (padrão 10; usar 1 para teste rápido), `N_JOBS` (processos em paralelo) e `SEEDS`. Por que: tudo que depende de ambiente fica num só lugar, e o padrão é o do artigo (10 repetições).

**A4 (células 5–7) — Análise exploratória.**
- Célula 6 (código): lê o CSV, mostra dimensões e valores faltantes; recorta treino/validação/teste do mesmo jeito que o `Dataset` (60/20/20 sequencial) e tabela a % de positivos em cada parte; salva o gráfico `rep_01_distribuicao_classes`.
- Célula 7 (markdown): leitura. Só ~6,2% são positivos (desbalanceamento de ≈ 15:1), e a proporção é parecida nas três partes.
- Por que existe: conhecer a base antes de modelar. **O desbalanceamento descoberto aqui é a base de toda a melhoria.**

**A5 (células 7–9) — Baselines.**
- Célula 8 (código): treina `DecisionTreeClassifier(random_state=0)` com todos os atributos, exatamente como no `Example.ipynb`, e compara com um classificador "burro" que sempre responde "renda ≤ 50K". Mostra acurácia, acurácia balanceada, precisão, recall e F1 de cada um.
- Célula 9 (markdown): a árvore com todos os atributos dá **92,815%** (artigo: 92,87%); o classificador de classe majoritária acerta **93,70%**, ou seja, mais que a própria árvore. O recall da árvore completa é ≈ 0,47 e o F1 ≈ 0,45.
- Por que existe: a DT é o *Baseline* da Tabela 2 (checagem do pipeline) e o majoritário é a "régua de sanidade" que revela quanto a acurácia vale nesta base.
- **[Observação honesta]** A diferença de 0,055 pp frente ao artigo não foi explicada. Testamos scikit-learn 1.6 a 1.9 e o resultado é idêntico, então não é versão da biblioteca. Hipótese não confirmada: desempate de *splits* da árvore em outro ambiente.

**A6 (células 9–10) — Execução dos 4 algoritmos.**
- Célula 10 (código): `runner.run_many(['original'], ALGOS, SEEDS, n_jobs=N_JOBS)`. Roda CHC_QX, PSO_QX, CHC e PSO com as sementes 0 a 9, em paralelo, gravando cada execução. Retorna a tabela de resultados.
- Por que existe: algoritmos estocásticos precisam de repetições; a gravação incremental permite retomar.

**A7 (células 11–13) — Resultados no formato da Tabela 2.**
- Célula 12 (código): monta a tabela "nossa mediana ± dp × artigo × Δ em pp × dentro da tolerância?".
- Critério de sucesso: a nossa mediana fica a até 1 desvio padrão do artigo + 0,1 pp.
- Por que existe: é o critério objetivo de sucesso da replicação.

**A8 (células 14–17) — Gráficos da replicação.**
- Célula 14: boxplots de acurácia (com a mediana do artigo em laranja, a árvore completa tracejada e a classe majoritária pontilhada) e do tempo até a melhor solução (escala log). Salva `rep_02_acuracia_e_tempo`.
- Célula 16: frequência com que cada atributo é escolhido nas soluções finais (barras e *heatmap* por algoritmo). Salva `rep_03_frequencia_atributos`.
- Por que existem: o primeiro mostra qualidade e a contribuição central do artigo (tempo); o segundo mostra *quais* atributos importam. Pontos de atenção: a coluna 24 é o *instance weight* do UCI.

**A9 (células 17–19) — Divergências e análise crítica.**
- Célula 17–18: tabela das divergências (q, tempo do CHC, shuffle) e uma tabela completa de métricas por algoritmo (acurácia, acurácia balanceada, precisão, recall, F1, MCC, ROC-AUC, nº de atributos).
- Célula 18 (código): gráfico "acurácia alta, recall baixo" (cada execução, com a árvore completa como estrela e a linha da classe majoritária) e a matriz de confusão mediana. Salva `rep_04_acuracia_vs_recall`.
- Célula 19 (markdown): **o ponto central.** As soluções do artigo (~94,9%) ficam só ~1,2 pp acima do classificador burro; o **recall é ≈ 0,24** (encontram 1 em cada 4 pessoas de renda alta), a precisão é ≈ 0,83, a acurácia balanceada é ≈ 0,62 e o F1 ≈ 0,37. Pior: a árvore com todos os atributos tinha recall 0,47 e F1 0,45, ou seja, **a busca por acurácia piorou o recall**. Cita a limitação reconhecida pelos autores.
- Por que existe: define o rumo da Parte B.

**Célula 20 (código) — Salvar CSVs da replicação.** Gera `results/replicacao_runs.csv` e `results/replicacao_resumo.csv`.

**Célula 21 — Cabeçalho da Parte B.** Mesma base, mesma divisão, mesmos 4 algoritmos, mesmos hiperparâmetros. Só muda o que a busca otimiza.

**B1 (célula 22) — Motivação.** Resume o problema encontrado (desbalanceamento, majoritário ≈ 93,7%, recall 0,24) e cita o trecho do artigo. Acrescenta a observação sobre a coluna 24 (UCI documenta que é o *instance weight*, que não deveria ser atributo de entrada).

**B1.5 (célula 23) — A melhoria em palavras simples.** Problema, melhoria e analogia (vigia de segurança avaliado só pela % de dias sem incidente). Explica por que a implementação por subclasse é elegante.

**B2 (células 24–27) — A modificação.**
- Célula 24: explica que `Evolution.py` sempre chama `set_validation_accuracy()` e `get_validation_accuracy()`, então sobrescrever esses métodos muda a fitness dos 4 algoritmos **e** o critério do meta-modelo do QX.
- Célula 25 (código): parâmetros da Parte B (`N_RUNS_CHC`, `RODAR_M3`, `RODAR_ABLACOES`, `RODAR_EXTRA`, `EXECUTAR`, `EXECUTAR_TRANSF`) e impressão do código de `DatasetMod.py`.
- Célula 26 (markdown): tabela das 9 configurações e a justificativa de cada.
- Célula 27 (código): monta a lista de configurações e roda todas (`PSO`, `PSO_QX` e `CHC`, `CHC_QX`) com `run_many`, em paralelo, com cache.
- Por que existe: isolar a mudança garante que só a fitness muda.

**B3 (células 28–30) — Tabela de resultados.** Célula 29: para cada algoritmo, tabela com mediana ± dp de acurácia, acurácia balanceada, precisão, recall, F1, MCC, ROC-AUC, nº de atributos e tempo, por configuração. Célula 30: explica como ler (acurácia tende a cair; recall, F1 e acurácia balanceada tendem a subir).

**B4 (células 30–31) — Testes estatísticos.** Célula 31: testes de Wilcoxon e *t* pareado por semente, `original` × cada modificação e comparações que isolam ingredientes (M2b × M2, M1 × M3, original × M3b, M1 × M4, original × X1, M1 × X2). Marca com `*` quando p < 0,05. Por que: diferenças podem ser acaso. Com n = 10, o menor p do Wilcoxon é ≈ 0,002.

**B5 (células 32–41) — Gráficos da melhoria.**
- B5.1 (célula 33): precisão × recall de cada execução, um painel por algoritmo. Salva `mel_01_precisao_vs_recall`. Mostra o trade-off.
- B5.2 (célula 35): boxplots de F1, acurácia balanceada e recall. Salva `mel_02_boxplots_metricas`.
- B5.3 (célula 37): matrizes de confusão da execução mediana de cada configuração (escolha automática, sem olhar qual é a melhor). Salva `mel_03_*`.
- B5.4 (célula 39): nº de atributos e tempo. Salva `mel_04_atributos_e_tempo`.
- B5.5 (célula 41): frequência dos atributos por configuração e **estabilidade de Jaccard** entre sementes. Salva `mel_05_frequencia_atributos`.

**B6 (células 42–44) — Ablações, MCC e coluna 24.** Explica a pergunta que cada configuração extra responde e mostra tabelas-resumo (mediana por configuração × algoritmo) e o top de atributos por configuração.

**B7 (células 45–46) — Transferência.** Pega os subconjuntos já escolhidos pela árvore e os avalia com Regressão Logística e Random Forest (sem nova busca), mais a referência com os 41 atributos. Mede ROC-AUC, PR-AUC, recall e F1 no teste. Salva `mel_06_transferencia_pr_auc`. Por que: se os atributos escolhidos também ajudam outros modelos, a melhoria é sobre os atributos e não um artefato da árvore.

**B8 (células 47–49) — Discussão e recomendação.** Célula 48 imprime, algoritmo a algoritmo, "acurácia antes → depois, recall antes → depois, precisão, F1, acurácia balanceada" para cada configuração. Célula 49 traz a leitura dos 10 resultados, a tabela "quando usar cada configuração", a recomendação final e as limitações.

**Célula 50 (código) — Salvar CSVs da melhoria.** `melhoria_runs.csv`, `melhoria_resumo.csv`, `melhoria_testes_estatisticos.csv`, `melhoria_transferencia.csv`.

**Parte C (célula 51) — Resumo para o relatório.** Responde ao roteiro da disciplina: descrição da base, tamanho da população, % treino/validação/teste, validação cruzada (não usada), algoritmos, critério dos hiperparâmetros, transfer learning/*dropout* (não se aplicam), aumento de dados (só em M3), desbalanceamento, métricas, resultado da replicação e da melhoria.

---

## PARTE 3 — Nossa melhoria

### 3.1 O problema, em palavras simples
- Só **6,2%** das pessoas da base ganham mais de US$ 50 mil.
- Um modelo que sempre diz "ganha pouco" já acerta **93,7%** (e não encontra nenhum rico).
- Os algoritmos do artigo maximizam **acurácia**. Eles são recompensados por acertar a maioria e acabam ignorando a classe rara.
- Resultado: subconjuntos que acertam **94,9%**, mas encontram só **24%** das pessoas ricas (recall 0,24). De cada 4 ricos, 3 passam despercebidos. E a árvore com os 41 atributos encontrava 47%.
- **Analogia:** avaliar um vigia só pela % de dias "sem incidente". Se 94% dos dias são tranquilos, o vigia que nunca dispara o alarme tira nota 94. Mudar a nota para "quantos incidentes reais detectou, sem alarmes falsos demais" muda quem é o melhor vigia.

### 3.2 A ideia
**Trocar só a "nota" (a *fitness*) que a busca maximiza.** Em vez de acurácia, usar uma métrica que só dá nota alta se o modelo realmente encontrar a classe positiva. Todo o resto fica igual: base, divisão, algoritmos, hiperparâmetros, classificador.

### 3.3 Como foi implementado (`melhoria/DatasetMod.py`)
- `Evolution.py` **sempre** pede a fitness chamando `set_validation_accuracy()` e `get_validation_accuracy()` da classe `Dataset`.
- Criamos a subclasse `DatasetMod(Dataset)` que sobrescreve `set_validation_accuracy()` e `set_test_accuracy()` para usar uma métrica escolhida (`accuracy`, `balanced_accuracy`, `f1` ou `mcc`).
- **Consequências:**
  1. A fitness dos **quatro** algoritmos (CHC, PSO, CHC_QX, PSO_QX) muda de uma só vez, sem editar `Evolution.py`.
  2. O **meta-modelo do QX também muda**: a correlação de Spearman passa a ser calculada sob a nova métrica, ou seja, o meta-modelo agora aprende a imitar o ranking de F1 (ou outra) e não o de acurácia. Isso é coerente com o argumento do artigo.
- O F1 usa `zero_division=0`: se a árvore não prevê nenhum positivo, F1 = 0.
- Extras opcionais na mesma classe:
  - `smote=True`: depois da divisão e da normalização, aplica SMOTE **só no treino** (com cache por processo, porque o treino é determinístico) e refaz a seleção de instâncias.
  - `drop_cols=[24]`: remove colunas por rótulo antes da divisão (idempotente).

### 3.4 As 9 configurações e o porquê de cada uma

| Config | Fitness | Classificador | Papel |
|---|---|---|---|
| `original` | acurácia | árvore padrão | O artigo, sem mudanças |
| `M1_f1` | **F1 da classe 1** | árvore padrão | **A melhoria principal**: troca só a métrica |
| `M2_bacc_bal` | acurácia balanceada (= recall médio, a sugestão dos autores) | árvore com `class_weight='balanced'` | Métrica + classificador sensível ao custo (erros na classe rara pesam ~15×) |
| `M3_f1_smote` | F1 | árvore padrão, treino com SMOTE | Aumento de dados sintético só no treino |
| `M2b_bacc` *(ablação)* | acurácia balanceada | árvore padrão, sem pesos | Isola o efeito da métrica do efeito do `class_weight` |
| `M3b_acc_smote` *(ablação)* | acurácia | árvore padrão, com SMOTE | Isola o efeito do SMOTE do efeito do F1 |
| `M4_mcc` | MCC (Matthews) | árvore padrão | Outra métrica balanceada |
| `X1_acc_sem24` | acurácia | árvore padrão, sem coluna 24 | Efeito de remover o *instance weight* |
| `X2_f1_sem24` | F1 | árvore padrão, sem coluna 24 | O mesmo, sob F1 |

**Por que cada métrica:**
- **F1** é a média harmônica de precisão e recall da classe positiva. Se o modelo nunca prevê positivos, F1 = 0.
- **Acurácia balanceada** = média dos recalls das duas classes. É a sugestão dos próprios autores ("average recall").
- **MCC** resume a matriz de confusão inteira e também é robusto a desbalanceamento.
- **SMOTE** cria exemplos sintéticos da classe rara interpolando vizinhos. **[Observação honesta]** A base é toda codificada em números; interpolar entre códigos de categoria não tem significado estrito, é só uma aproximação.
- **Regras para evitar vazamento:** SMOTE só no treino, depois do split e da normalização; validação e teste ficam com a distribuição real. As métricas de fitness foram escolhidas por justificativa teórica, **nunca** olhando o desempenho no teste.

### 3.5 Avaliação de transferência
Para ver se a melhoria é sobre os **atributos** e não um artefato da árvore, treinamos Regressão Logística (balanceada) e Random Forest (30 árvores, balanceada) no treino original, usando apenas os atributos escolhidos por cada configuração, e medimos no teste ROC-AUC, PR-AUC, recall e F1.

---

## PARTE 4 — Resultados (10 sementes; medianas no teste)

### 4.1 Replicação: nosso resultado × artigo (acurácia, %)

| Método | Nossa mediana ± dp | Artigo (mediana ± dp) | Δ (pp) |
|---|---|---|---|
| Baseline DT (41 atributos) | 92,815 | 92,87 | −0,05 |
| CHC | 94,92 ± 0,02 | 94,63 ± 0,35 | +0,29 |
| CHC_QX | 94,93 ± 0,02 | 94,94 ± 0,07 | −0,01 |
| PSO | 94,91 ± 0,09 | 94,79 ± 0,24 | +0,12 |
| PSO_QX | 94,95 ± 0,05 | 94,94 ± 0,06 | +0,01 |

(No notebook, todas as linhas ficam "dentro da tolerância": até 1 dp do artigo + 0,1 pp. As médias, e não as medianas, do PSO e do PSO_QX são 94,87 e 94,93, porque o PSO puro tem algumas execuções piores.)

**Veredito:** replicação bem-sucedida.
- As **medianas dos QX batem com o artigo a até 0,01 pp**.
- Os puros (CHC e PSO) saem **um pouco acima** do artigo e com variação menor.
- Todos os algoritmos superam a árvore com 41 atributos em cerca de 2 pp, como o artigo.
- O desvio padrão do PSO_QX (0,05) é menor que o do PSO (0,09); no CHC os dois têm 0,02 (empate). O artigo também reporta menor variância nos QX; aqui isso só aparece no PSO.
- **Tempo:** os QX são **3 a 4× mais rápidos** (tempo total mediano por execução: original ≈ 70–75 s nos QX contra ≈ 230–270 s nos puros; pelo tempo até a melhor solução, ≈ 3,4–3,6×). Os tempos foram medidos com 8 processos simultâneos numa máquina de 10 núcleos, então valem como ordem de grandeza. Essa é a contribuição central do artigo, e ela se reproduziu.
- Os subconjuntos escolhidos são pequenos: mediana de 5 a 8 atributos dos 41.
- **[Observações honestas]** (a) A baseline deu 92,815% contra 92,87%, causa não identificada. (b) CHC e PSO puros ficam acima do artigo e com menor dispersão; não investigamos a causa (as divergências q e orçamento de tempo do CHC são candidatas, mas não testadas isoladamente). (c) O artigo não informa as sementes, então a comparação é estatística, não bit a bit.

### 4.2 O diagnóstico (o que a acurácia esconde)
Com a configuração original: acurácia ≈ 94,9%, mas **recall ≈ 0,24, precisão ≈ 0,83, F1 ≈ 0,375, acurácia balanceada ≈ 0,62**. A árvore com 41 atributos tinha recall 0,47 e F1 0,45.

### 4.3 Melhoria: medianas no teste, por configuração e algoritmo

| Config | Acurácia | Recall | F1 | MCC | Nº atributos (mediana) |
|---|---|---|---|---|---|
| original (4 algoritmos) | 94,9% | 0,24 | 0,375–0,379 | 0,43 | 5–8 |
| **M1_f1** | 93,0–93,2% | 0,47–0,48 | 0,465–0,468 | 0,43 | 25–33 |
| M2_bacc_bal | 81–82,5% | 0,85–0,87 | 0,36–0,38 | 0,38–0,40 | 6–7,5 |
| M3_f1_smote | 92,1–92,6% | 0,49–0,51 | 0,446–0,449 | 0,41 | 22–34 |
| M2b_bacc | 93,0–93,1% | 0,48–0,49 | 0,465–0,471 | 0,43 | 27–33 |
| M3b_acc_smote | 94,3–94,7% | 0,20–0,27 | 0,31–0,40 | 0,35–0,42 | 4–5 (varia de 1 a 13) |
| M4_mcc | 93,0–93,6% | 0,45–0,49 | 0,465–0,474 | 0,43–0,44 | 23–33 |
| X1_acc_sem24 | 94,9% | 0,24 | 0,375–0,377 | 0,43–0,44 | 5–8 |
| X2_f1_sem24 | 93,1–93,2% | 0,47–0,49 | 0,467–0,470 | 0,43 | 23–35 |

(Valores são intervalos entre as medianas dos 4 algoritmos: CHC, CHC_QX, PSO, PSO_QX. Atenção: o **MCC não muda** com M1; a precisão do original varia de 0,79 (PSO) a 0,84.)

**Valores detalhados do PSO_QX (algoritmo-foco), original → M1:** acurácia 94,95% → 93,1%; recall 0,243 → 0,48 (Δ ≈ +0,24, Wilcoxon p ≈ 0,002); precisão ≈ 0,84 → ≈ 0,46; F1 ≈ 0,377 → ≈ 0,465; acurácia balanceada 0,62 → 0,72; acurácia cai ≈ 1,9 pp (p ≈ 0,002).

### 4.4 Conclusões por configuração

1. **A acurácia cai quando trocamos a fitness — e isso é esperado.** O original otimiza exatamente a acurácia. M1, M3 e M4 ficam abaixo do classificador majoritário (93,7%), o que só preocupa se o objetivo for acertar o máximo de pessoas; aqui o objetivo é encontrar quem tem renda alta.
2. **M1 (F1) é o melhor equilíbrio.** Nos quatro algoritmos, o recall sobe de ≈ 0,24 para ≈ 0,47–0,48 (Δ ≈ +0,23, p ≈ 0,002 em todos), o F1 de ≈ 0,376 para ≈ 0,466 (Δ ≈ +0,09, p ≈ 0,002) e a acurácia balanceada de ≈ 0,62 para ≈ 0,72 (p ≈ 0,002). A precisão cai de ≈ 0,79–0,84 para ≈ 0,45–0,46. **O MCC, porém, não melhora**: fica em ≈ 0,43 (Δ entre −0,009 e 0, p ≥ 0,08) — o ganho de recall é pago, quase na mesma medida, com perda de precisão. O F1 fica acima do da árvore com os 41 atributos (≈ 0,45), sem alterar o classificador.
3. **M2 (acur. balanceada + `class_weight`) vai ao extremo do recall.** Recall ≈ 0,85–0,87 e acurácia balanceada ≈ 0,84 (a maior de todas), mas a precisão cai para ≈ 0,23–0,25 e o MCC cai (≈ 0,38–0,40, p ≈ 0,002 em todos). O F1 (≈ 0,36–0,38) fica **próximo do original**: menor nos PSO (Δ ≈ −0,013 a −0,017, p < 0,02), sem diferença significativa nos CHC. A ablação **M2b** (mesma acurácia balanceada **sem** `class_weight`) cai no mesmo ponto de M1 (recall ≈ 0,48, F1 ≈ 0,47): o comportamento extremo de M2 vem do **`class_weight`**, não da métrica (F1 de M2b − M2 ≈ +0,08 a +0,11, p ≈ 0,002 nos 4). Faz sentido quando **perder um positivo custa muito mais que um alarme falso**.
4. **M3 (F1 + SMOTE) não compensou.** Recall um pouco maior que o de M1 (≈ 0,50 contra ≈ 0,48; p < 0,03 nos 4), mas F1 e MCC menores nos quatro algoritmos (F1 ≈ 0,446 contra ≈ 0,466; Δ ≈ −0,02; significativo no CHC, p ≈ 0,002, e no PSO pelo teste *t*, mas não no CHC_QX e PSO_QX, p ≈ 0,08–0,38), acurácia menor e custo de tempo ≈ 3,5× (≈ 3.100–3.350 s contra ≈ 860 s nos puros). **M3b** (acurácia + SMOTE) **não ajuda e é instável**: recall e F1 sem ganho significativo no PSO/QX (p ≥ 0,27), *piores* no CHC (F1 −0,06, p ≈ 0,049) e com execuções degeneradas (no CHC, 2 de 10 execuções acertaram a maioria mas encontraram ≈ 0 % dos positivos). O SMOTE sozinho não faz a busca olhar para a classe rara; quem faz isso é a métrica.
5. **MCC (M4) ≈ F1 (M1).** Recall e F1 de M4 não diferem de M1 de forma significativa (recall p ≥ 0,09; F1 p ≥ 0,06). Note que otimizar o MCC **não aumentou o MCC** em relação ao original (≈ 0,43 nos dois; Δ entre −0,009 e +0,008): a base tem um ótimo de MCC já alcançado pela acurácia, o que reforça que MCC e F1 medem trade-offs diferentes — o MCC "não enxerga" o ganho de recall que o F1 enxerga. Para encontrar a classe rara, F1 é a escolha mais simples de explicar.
6. **A coluna 24 (instance weight) não muda nada.** Removê-la (X1, X2) deixa acurácia, recall e F1 iguais aos de `original` e M1 (recall e F1: p ≥ 0,37; Δ ≈ 0). A busca não dependia dela.
7. **O ganho de tempo do QX se mantém.** Tempo total mediano por execução (inclui carga dos dados e reavaliação final; medido com 8 processos em paralelo, então vale como ordem de grandeza): original ≈ 70–75 s (QX) contra ≈ 230–270 s (puros); com F1 ≈ 210–280 s (QX) contra ≈ 860 s (puros). Pelo tempo até a melhor solução (`time_best`), o ganho é ≈ 3,4–3,6× no original e ≈ 2,4–7× com F1. O custo absoluto do F1 é maior que o da acurácia (≈ 3–4× nos QX).
8. **Os subconjuntos mudam.** Original: 5–8 atributos. F1/MCC: ≈ 23–35. M2: ≈ 6–7. Acurácia + SMOTE: ≈ 4–5. Enxergar a classe rara exige mais informação do que acertar a maioria.

### 4.5 Transferência (LR e RF treinados só com os atributos escolhidos; medianas no teste)

| Config | LR ROC-AUC | LR PR-AUC | RF ROC-AUC | RF PR-AUC |
|---|---|---|---|---|
| original | 0,799 | 0,345 | 0,872 | 0,406 |
| M1_f1 | 0,925 | 0,536 | 0,941 | 0,615 |
| M2_bacc_bal | 0,855 | 0,352 | 0,918 | 0,519 |
| M3_f1_smote | 0,927 | 0,542 | 0,942 | 0,617 |
| M2b_bacc | 0,927 | 0,542 | 0,942 | 0,618 |
| M3b_acc_smote | 0,749 | 0,270 | 0,835 | 0,354 |
| M4_mcc | 0,925 | 0,534 | 0,941 | 0,613 |
| X1_acc_sem24 | 0,799 | 0,340 | 0,857 | 0,396 |
| X2_f1_sem24 | 0,926 | 0,542 | 0,942 | 0,615 |

**Leitura:** os subconjuntos escolhidos com F1/MCC (e M2b, M3) dão PR-AUC ≈ 0,53–0,54 (LR) e ≈ 0,61–0,62 (RF), contra ≈ 0,35 e ≈ 0,41 dos subconjuntos do original. Ou seja, **os atributos escolhidos pela nova fitness carregam informação útil também para outros modelos**. Os subconjuntos do original (≈ 6 atributos) e de M3b (4) são os piores; M2 (≈ 7) fica no meio.
**[Observação honesta]** A diferença de nº de atributos (≈ 29 vs ≈ 6) é parte da explicação: subconjuntos maiores naturalmente ajudam mais outros modelos. Não fizemos um controle com o mesmo nº de atributos.

### 4.6 Comparação com o artigo original — resumo
| Aspecto | Artigo | Nossa replicação | Nossa melhoria |
|---|---|---|---|
| Acurácia (QX) | ≈ 94,94% | ≈ 94,93% | ≈ 93,1% (F1) |
| Recall da classe positiva | não reportado | ≈ 0,24 (descoberto por nós) | ≈ 0,48 (dobra) |
| Tempo QX × puro | QX bem mais rápido | 3–4× mais rápido | 3–4× mais rápido (mantém) |
| Nº de atributos | pequeno | 5–8 | 25–33 (F1) |
| Desbalanceamento | reconhecido como limitação | confirmado: acurácia esconde o problema | tratado trocando a fitness |

---

## PARTE 5 — Recomendação e limitações

### 5.1 Quando usar cada configuração
| Situação | Recomendação |
|---|---|
| Classes balanceadas, ou acurácia é de fato o objetivo | `original` |
| Compromisso entre achar positivos e não gerar alarmes falsos demais (caso geral) | **M1 (F1)** — M4 (MCC) e M2b são equivalentes |
| Perder um positivo custa muito mais que um alarme falso | M2 (acurácia balanceada + `class_weight`) |
| Interesse em aumento de dados | M3 — sem vantagem clara e ≈ 3,5× mais lento |

**Recomendação final:** usar **F1 como fitness** no lugar da acurácia (se o objetivo for encontrar a classe positiva; se o critério for o MCC, a acurácia original já é equivalente). É uma troca de uma linha, o recall dobra, F1 e acurácia balanceada sobem e o ganho de velocidade do QX permanece. O custo é perder ≈ 1,8 pp de acurácia, precisão menor, subconjuntos maiores (≈ 25–33 atributos em vez de 5–8) e mais tempo de busca. **A escolha entre precisão e recall é uma decisão de negócio, não estatística.**

### 5.2 Limitações (devem aparecer na apresentação)
- A divisão 60/20/20 é **única** e sequencial (sem validação cruzada); a variabilidade vem só das sementes das buscas.
- A melhor solução é escolhida na validação; existe alguma tendência de *overfitting* à validação (por isso o teste só é usado no fim).
- Árvore de decisão totalmente crescida com limiar fixo (0,5): outros classificadores ou ajuste do limiar poderiam mudar o trade-off.
- Com n = 10 sementes, o Wilcoxon tem piso de p ≈ 0,002; diferenças pequenas (como F1 de M1 × M3) não são detectáveis.
- O SMOTE em atributos codificados é só uma aproximação.
- Os tempos foram medidos com 8 processos em paralelo numa máquina de 10 núcleos: valem como ordem de grandeza.
- A configuração M3b (acurácia + SMOTE) é instável: algumas execuções do CHC encontraram ≈ 0% dos positivos.
- A baseline deu 92,815% contra 92,87%, com causa não identificada.
- O artigo não informa as sementes; a comparação com ele é estatística.

### 5.3 Perguntas prováveis da banca e respostas curtas
- **"Por que a acurácia caiu e isso é bom?"** Porque o original maximiza acurácia, que na base recompensa ignorar a classe rara. Trocamos 1,8 pp de acurácia por o dobro de ricos encontrados.
- **"Por que F1 e não acurácia balanceada?"** Acurácia balanceada pura (M2b) dá resultado equivalente; o que torna o M2 extremo é o `class_weight`. F1 equilibra precisão e recall e é fácil de explicar.
- **"O SMOTE ajuda?"** Não nesta base: não melhora o F1 e é ≈ 3,5× mais lento.
- **"A coluna 24 inflava o resultado?"** Não: removê-la não alterou nada.
- **"Funciona para outro modelo?"** Os atributos escolhidos com F1 aumentam bastante o PR-AUC de Regressão Logística e Random Forest (com a ressalva de que são subconjuntos maiores).
- **"Vocês mudaram o código dos autores?"** Só 3 correções de compatibilidade, documentadas. A melhoria vive numa subclasse separada.
- **"O MCC melhorou?"** Não: ficou em ≈ 0,43 tanto no original quanto no M1 (e no M4, que otimiza o MCC). O ganho de recall é compensado por perda de precisão; F1 e acurácia balanceada mostram o ganho, o MCC não.
- **"Como garantem que não usaram o teste para escolher?"** Fitness sempre na validação; configurações e métricas escolhidas por justificativa antes de ver o teste; nenhuma busca de hiperparâmetros.

---

## PARTE 6 — Sugestão de estrutura da apresentação (opcional)

1. Contexto e problema (seleção de atributos; por que é cara).
2. A ideia do artigo (CHC, PSO e o QX).
3. Como replicamos (regras, hiperparâmetros, 10 sementes, 3 correções).
4. Resultado da replicação (tabela nossa × artigo; figura `rep_02`; veredito).
5. O achado: a acurácia esconde o problema (figuras `rep_01` e `rep_04`).
6. Nossa melhoria (troca da fitness, `DatasetMod`, analogia do vigia).
7. Resultados da melhoria (figuras `mel_01`, `mel_02`, `mel_03`; tabela M1 × original).
8. Ablações (M2b, M3b, MCC, coluna 24) e transferência (`mel_06`).
9. Recomendação, trade-offs e limitações.
10. Conclusão e perguntas.

### Figuras disponíveis em `nossa_replica/figures/`
`rep_01_distribuicao_classes`, `rep_02_acuracia_e_tempo`, `rep_03_frequencia_atributos`, `rep_04_acuracia_vs_recall`, `mel_01_precisao_vs_recall`, `mel_02_boxplots_metricas`, `mel_03_matrizes_confusao_PSO_QX` (e as versões dos outros 3 algoritmos), `mel_04_atributos_e_tempo`, `mel_05_frequencia_atributos`, `mel_06_transferencia_pr_auc`.

### Arquivos de dados em `nossa_replica/results/`
`replicacao_runs.csv`, `replicacao_resumo.csv`, `melhoria_runs.csv`, `melhoria_resumo.csv`, `melhoria_testes_estatisticos.csv`, `melhoria_transferencia.csv`, e `runs/` (um JSON por execução).
