from nbhelp import *
nb = novo()

md(nb, r"""
# Notebook 1 — Replicação do artigo *Fast Genetic Algorithm for feature selection — A qualitative approximation approach*

**Disciplina:** IA Avançada (UNICAP) · **Base:** census-income (UCI) · **Artigo:** Altarabichi, Nowaczyk, Pashami & Mashhadi, *Expert Systems with Applications* 211 (2023) 118528 · [arXiv:2404.03996](https://arxiv.org/pdf/2404.03996) · **Código oficial:** [Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection](https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection)

O objetivo deste notebook é **replicar exatamente** o experimento do artigo na base census-income, usando o código dos autores sem alterar sua lógica, e depois **examinar criticamente** os resultados — o que prepara a melhoria do Notebook 2.

## 1. O problema e a ideia do artigo

**Seleção de atributos (*feature selection*) por *wrapper*.** Dada uma base com 41 atributos, queremos o subconjunto que maximiza o desempenho de um classificador (aqui, uma Árvore de Decisão). Um *wrapper* avalia cada subconjunto candidato **treinando o classificador** — preciso, porém caro: com 41 atributos há $2^{41}$ subconjuntos e cada avaliação treina um modelo com ~120 mil instâncias.

**Algoritmos genéticos / enxames.** Em vez de testar todos, uma meta-heurística evolui uma população de subconjuntos (vetores binários de 41 posições: 1 = atributo selecionado). Cada indivíduo precisa de uma avaliação de *fitness*, e isso domina o tempo de execução.

**A ideia do artigo — aproximação qualitativa (QX).** Treinar com **menos instâncias** é mais barato, e o valor absoluto da acurácia muda, mas o que a busca realmente precisa é do **ranking** dos subconjuntos (qual é melhor que qual). Então:

1. Sorteia-se um conjunto de `q` subconjuntos de atributos aleatórios e avalia-se cada um com a **função original** (treino completo).
2. Busca-se uma **amostra de instâncias** tal que o modelo treinado nela produza um ranking *parecido* com o original. A semelhança é medida pela **correlação de postos de Spearman** entre as acurácias aproximadas e as originais; a amostra também é penalizada pelo tempo que gasta (`distância = (1 − ρ) + tempo_aprox / tempo_original`).
3. O **meta-modelo** (treino na amostra escolhida) passa a ser a fitness do algoritmo evolutivo.
4. Como a aproximação pode errar, há **controle de evolução**: a cada `f` gerações os melhores indivíduos são reavaliados com a função original; a busca só termina após `f_no_change` rodadas sem melhoria.

## 2. Os quatro algoritmos comparados

| Algoritmo | O que é | Parâmetros no notebook oficial |
|---|---|---|
| **CHC_QX** | CHC (algoritmo genético com seleção elitista, cruzamento HUX e reinício por *cataclysmic mutation*) + meta-modelo QX | `f=10`, `q=10`, `f_no_change=2`, população 50 |
| **PSO_QX** | PSO (enxame de partículas, topologia estrela) + meta-modelo QX | `f=10`, `q=10`, `f_no_change=2`, 50 partículas |
| **CHC** | CHC puro, fitness com função original | população 50, `d = k//4`, `max_no_change=10` |
| **PSO** | PSO puro, fitness com função original | 50 partículas, `steps_no_change=10` |

Em todos, **a fitness é a acurácia no conjunto de validação**; o resultado final é reportado como **acurácia no conjunto de teste**. O teste nunca participa da busca.
""")

md(nb, r"""
## 3. Instalação e importações

Dependências: `deap`, `pyswarms`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `joblib` (e `imbalanced-learn` para o Notebook 2). Veja o `README.md` para o passo a passo (local e Colab).

### Correções de compatibilidade (as **únicas** alterações no código dos autores)

O código foi escrito em 2022 e quebra com numpy 2 / pandas recentes. Foram feitas exatamente três correções, **sem tocar na lógica**:

| Arquivo | Original | Corrigido | Por quê |
|---|---|---|---|
| `code/Dataset.py` | `np.NaN` | `np.nan` | `np.NaN` foi removido no numpy 2 |
| `code/Dataset.py` | `pd.read_csv(..., delim_whitespace=True, ...)` | `pd.read_csv(..., sep=r"\s+", ...)` | `delim_whitespace` foi descontinuado no pandas. Esse ramo **não é usado** pelo census-income (que é separado por vírgula) |
| `code/Example.ipynb` | `np.warnings.filterwarnings(...)` | linha removida | `np.warnings` não existe mais; a linha apenas silenciava um aviso |

`Evolution.py` **não foi alterado**.

### Reprodutibilidade e paralelismo

* **Sementes:** cada execução `s ∈ {0,…,9}` chama `random.seed(s)` e `np.random.seed(s)` (o `pyswarms` usa `np.random`). Assim, cada execução é determinística.
* **`OMP_NUM_THREADS=1`:** cada processo usa uma única thread; o paralelismo vem de rodar execuções independentes em processos separados (`joblib`). Isso também mantém a medição de tempo comparável entre execuções.
* **Retomada:** cada execução é gravada em `results/runs/*.json` assim que termina. Se o kernel cair, basta reexecutar a célula — só o que falta é refeito.
""")

code(nb, r"""
import os, sys, warnings
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'          # uma thread por processo (ver explicação acima)
warnings.filterwarnings('ignore')

from pathlib import Path
RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(RAIZ / 'src')); sys.path.insert(0, str(RAIZ / 'code'))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, precision_score

import runner                      # executor (src/runner.py): 1 execução = (config, algoritmo, semente)
from analise import *              # tabelas, testes, paleta e salvar_fig
from Dataset import Dataset        # código ORIGINAL dos autores

# ------------------------- PARÂMETROS -------------------------
N_RUNS = int(os.environ.get('N_RUNS', 10))   # nº de repetições por algoritmo (artigo: 10). Use 1 para teste rápido.
N_JOBS = int(os.environ.get('N_JOBS', max(1, min(8, (os.cpu_count() or 2) - 1))))   # processos em paralelo
SEEDS = list(range(N_RUNS))
# --------------------------------------------------------------
print(f'N_RUNS={N_RUNS}  N_JOBS={N_JOBS}  sementes={SEEDS}')
""")

md(nb, r"""
## 4. Análise exploratória da base

A base census-income (UCI, *Census-Income KDD*) tem como tarefa prever se a renda anual de uma pessoa **excede US$ 50 mil**. O arquivo já vem **numérico e codificado** (sem cabeçalho); o código dos autores aplica `LabelEncoder` a colunas textuais (não há nenhuma aqui) e preenche faltantes pela mediana (também não há). A **última coluna (índice 41) é o rótulo**.

**Divisão 60/20/20 sem embaralhar.** `Dataset.divide_dataset` usa os primeiros 60 % das linhas para treino, os 20 % seguintes para validação e os 20 % finais para teste, **na ordem do arquivo** (`shuffle=False`). A normalização *z-score* usa média e desvio **só do treino**, evitando vazamento de informação. Convém verificar se a proporção da classe positiva se mantém nas três partes, já que a divisão não é estratificada.
""")

code(nb, r"""
dados = pd.read_csv(RAIZ / 'data' / 'census-income.data', header=None)
y = dados.iloc[:, -1].values
print('Dimensões:', dados.shape, '(linhas × colunas, incluindo o rótulo)')
print('Valores faltantes:', int(dados.isna().sum().sum()), '| tipos:', dados.dtypes.unique())

n = len(dados); i1, i2 = int(.6 * n), int(.8 * n)
partes = {'treino (60%)': y[:i1], 'validação (20%)': y[i1:i2], 'teste (20%)': y[i2:], 'base toda': y}
tab = pd.DataFrame({k: {'instâncias': len(v), 'positivos (renda > 50K)': int(v.sum()),
                        '% positivos': round(100 * v.mean(), 2), '% classe majoritária': round(100 * (1 - v.mean()), 2)}
                    for k, v in partes.items()}).T
display(tab)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(list(partes)[:3], [100 * v.mean() for v in list(partes.values())[:3]], color=CORES['original'], width=.55)
ax.set_ylabel('% de positivos'); ax.set_title('Proporção da classe positiva em cada parte')
for i, v in enumerate(list(partes.values())[:3]): ax.text(i, 100 * v.mean() + .1, f'{100 * v.mean():.2f}%', ha='center')
ax.set_ylim(0, 8); salvar_fig(fig, 'rep_01_distribuicao_classes'); plt.show()
""")

md(nb, r"""
**Leitura.** Só ~6,2 % das instâncias são positivas: a base é **fortemente desbalanceada** (≈ 15,1 : 1). A proporção é muito parecida nas três partes, então a divisão sequencial não introduz um deslocamento relevante de classes. O desbalanceamento é a observação central deste trabalho — ele volta na seção de limitações e motiva o Notebook 2.

## 5. Baselines

* **Árvore de Decisão com todos os atributos** (`DecisionTreeClassifier(random_state=0)`) — é o *Baseline DT* da Tabela 2 do artigo (92,87 %).
* **Classificador de classe majoritária** — um modelo "burro" que sempre responde "renda ≤ 50K". Não aprende nada, mas serve de **régua de sanidade** para qualquer acurácia nesta base.
""")

code(nb, r"""
clf = DecisionTreeClassifier(random_state=0)
ds = Dataset(str(RAIZ / 'data' / 'census-income.data'), ',', -1, divide_dataset=False, header=None)
ds.divide_dataset(clf, normalize=True, shuffle=False, all_features=True, all_instances=True,
                  evaluate=True, partial_sample=False)       # exatamente como o Example.ipynb

pred_dt = ds.clf.predict(ds.X_test)
maj = np.zeros_like(ds.y_test)                                # sempre prevê a classe 0
def resumo(y_true, y_pred):
    return {'acurácia (%)': 100 * accuracy_score(y_true, y_pred),
            'acurácia balanceada': balanced_accuracy_score(y_true, y_pred),
            'precisão': precision_score(y_true, y_pred, zero_division=0),
            'recall (classe 1)': recall_score(y_true, y_pred, zero_division=0),
            'F1 (classe 1)': f1_score(y_true, y_pred, zero_division=0)}
base = pd.DataFrame({'Árvore, todos os atributos': resumo(ds.y_test, pred_dt),
                     'Classe majoritária': resumo(ds.y_test, maj)}).T
display(base.round(4))
print(f'\nAcurácia de teste da DT (todos os atributos): {100*ds.TestAccuracy:.3f}%   | valor do artigo: 92,87%')
""")

md(nb, r"""
**Leitura.**

* A Árvore de Decisão com todos os atributos fica **muito próxima** do valor do artigo (92,87 %). O valor que obtemos com o ambiente atual (numpy 2 / pandas 3 / scikit-learn 1.9) é conferido na célula acima; testamos as versões 1.6 a 1.9 do scikit-learn e o resultado é idêntico entre elas, então eventuais diferenças de centésimos de ponto em relação ao artigo não vêm da versão da biblioteca (provavelmente do desempate de *splits* da árvore em outro ambiente).
* A árvore completa tem acurácia baixa, mas **recall ≈ 0,47 e F1 ≈ 0,45** — bem melhor na classe de interesse do que as soluções que o GA/PSO encontram otimizando acurácia (seção 10).
* O classificador de **classe majoritária acerta ≈ 93,7 %**, ou seja, **mais que a própria árvore com todos os atributos**. Isso já indica que a acurácia, nesta base, é uma métrica enganosa: um modelo sem nenhuma capacidade preditiva "ganha" dela. O recall dele é zero — não encontra nenhum rico.

## 6. Execução dos quatro algoritmos

Cada célula abaixo roda **N_RUNS execuções por algoritmo** com os hiperparâmetros exatos do notebook oficial (ver `src/runner.py`):

```python
DecisionTreeClassifier(random_state=0)
divide_dataset(normalize=True, shuffle=False, all_features=True, all_instances=True, evaluate=True, partial_sample=False)
CHC_QX : Evolution.CHCqx(dataset, 10, 10, 2, 50)                      # f=10, q=10, f_no_change=2, população 50
PSO_QX : Evolution.PSOqx(dataset, options, 10, 10, 2, 50)             # options = {c1:1.49618, c2:1.49618, w:0.7298, k:50, p:2}
CHC    : Evolution.CHC(dataset, toolbox, d=41//4, pop50, max_no_change=10)
PSO    : Evolution.PSO(dataset, options, 50, steps_no_change=10)
```

Para cada solução final, o executor treina uma árvore com os atributos escolhidos e calcula **no teste**: acurácia, acurácia balanceada, precisão, recall, F1 (classe 1), MCC, ROC-AUC (`predict_proba`), matriz de confusão, nº de atributos, índices dos atributos e o **tempo até a melhor solução** (para os QX, inclui a construção do meta-modelo). O conjunto de teste só é usado nesta avaliação final.

> ⏱️ **Tempo.** CHC_QX e PSO_QX levam ~1 min por execução; CHC e PSO puros levam bem mais (dezenas de minutos, pois cada avaliação treina a árvore com todas as ~120 mil instâncias). Com `N_RUNS=10` e 8 processos, conte com algumas dezenas de minutos no total.
""")

code(nb, r"""
import time; t0 = time.time()
df = runner.run_many(['original'], runner.ALGOS, SEEDS, n_jobs=N_JOBS)
print(f'\nTempo desta célula: {(time.time()-t0)/60:.1f} min  |  execuções disponíveis: {len(df)}')
df.head()
""")

md(nb, r"""
## 7. Resultados no formato da Tabela 2 do artigo

Acurácia de teste (%), **mediana ± desvio padrão** das execuções, lado a lado com os valores do artigo e com a diferença (nossa mediana − artigo), em pontos percentuais.
""")

code(nb, r"""
linhas = []
baseline_pct = 100 * ds.TestAccuracy
linhas.append(['Baseline DT', f'{baseline_pct:.2f}', '92.87', f'{baseline_pct - 92.87:+.2f}', 'sim' if abs(baseline_pct - 92.87) <= 0.1 else 'não'])
for a in ['CHC', 'CHC_QX', 'PSO', 'PSO_QX']:
    s = 100 * df[df.algo == a]['test_accuracy']
    med, dp = s.median(), (s.std(ddof=1) if len(s) > 1 else np.nan)
    ref_m, ref_dp = ARTIGO[a]
    dentro = abs(med - ref_m) <= max(ref_dp, 0.1) * 1.0 + 0.1       # tolerância: 1 dp do artigo + 0,1 pp
    linhas.append([a, med_dp(df[df.algo == a]['test_accuracy']), f'{ref_m:.2f} ± {ref_dp:.2f}', f'{med - ref_m:+.2f}', 'sim' if dentro else 'não'])
tabela2 = pd.DataFrame(linhas, columns=['Algoritmo', 'Nossa replicação (mediana ± dp)', 'Artigo (mediana ± dp)', 'Δ mediana (pp)', 'Dentro da tolerância?'])
display(tabela2)
print('Execuções por algoritmo:', df.groupby('algo').size().to_dict())
""")

md(nb, r"""
**Critério de sucesso adotado:** consideramos a replicação bem-sucedida para um algoritmo se a nossa mediana ficar a até **1 desvio padrão do artigo + 0,1 ponto percentual** da mediana publicada (o 0,1 pp absorve a diferença de ambiente e de geração de números aleatórios — o artigo não informa as sementes).

**Discussão.** Os resultados esperados são: (i) todos os algoritmos de seleção de atributos superam a árvore com todos os atributos em ~2 pp; (ii) as variantes **QX** igualam ou superam as puras, com **desvio padrão menor** (a busca guiada por meta-modelo é mais estável) e (iii) QX chega lá bem mais rápido (gráfico de tempo, adiante). A coluna "Dentro da tolerância?" resume quão bem a replicação reproduziu a Tabela 2; os pontos em que ela não reproduz devem ser lidos junto com a seção de divergências (seção 9).

## 8. Gráficos
""")

code(nb, r"""
ordem = ['CHC', 'CHC_QX', 'PSO', 'PSO_QX']
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
d = df.assign(acc=100 * df.test_accuracy)
sns.boxplot(data=d, x='algo', y='acc', order=ordem, color=CORES['original'], width=.5, ax=axs[0], fliersize=0)
sns.stripplot(data=d, x='algo', y='acc', order=ordem, color='#222', size=4, ax=axs[0])
axs[0].axhline(100 * ds.TestAccuracy, ls='--', c='#888'); axs[0].text(-0.45, 100 * ds.TestAccuracy + .03, 'DT, todos os atributos', ha='left', fontsize=8, color='#555')
axs[0].axhline(100 * (1 - ds.y_test.mean()), ls=':', c='#C0392B'); axs[0].text(-0.45, 100 * (1 - ds.y_test.mean()) + .03, 'classe majoritária', ha='left', fontsize=8, color='#C0392B')
for i, a in enumerate(ordem):
    axs[0].plot([i - .3, i + .3], [ARTIGO[a][0]] * 2, c='#D55E00', lw=2, label='mediana do artigo' if i == 0 else None)
axs[0].set_xlabel(''); axs[0].set_ylabel('acurácia de teste (%)'); axs[0].set_title('Acurácia de teste por algoritmo'); axs[0].legend(loc='center right')

sns.boxplot(data=df, x='algo', y='time_best', order=ordem, color='#56B4E9', width=.5, ax=axs[1], fliersize=0)
sns.stripplot(data=df, x='algo', y='time_best', order=ordem, color='#222', size=4, ax=axs[1])
axs[1].set_yscale('log'); axs[1].set_xlabel(''); axs[1].set_ylabel('segundos (escala log)'); axs[1].set_title('Tempo até a melhor solução')
plt.tight_layout(); salvar_fig(fig, 'rep_02_acuracia_e_tempo'); plt.show()
print(df.groupby('algo')['time_best'].agg(['median', 'min', 'max']).round(1).loc[ordem])
""")

md(nb, r"""
**Leitura.** O primeiro painel compara a distribuição das acurácias com a mediana publicada (traço laranja), com a árvore completa (tracejado) e com a classe majoritária (pontilhado vermelho). O segundo mostra o ganho de tempo dos métodos QX (eixo logarítmico): é a contribuição central do artigo — **mesma qualidade, bem menos tempo**.

Em seguida, a frequência com que cada atributo é escolhido nas soluções finais. Atributos escolhidos por quase todas as execuções são os realmente informativos; os demais são ruído ou redundância.
""")

code(nb, r"""
freq = np.zeros((4, 41))
for i, a in enumerate(ordem):
    sub = df[df.algo == a]
    for fs in sub['features']:
        freq[i, fs] += 1 / len(sub)
fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={'height_ratios': [1, 1.25]})
geral = freq.mean(0) * 100
axs[0].bar(range(41), geral, color=CORES['original']); axs[0].set_ylabel('% das soluções'); axs[0].set_title('Frequência de seleção de cada atributo (média dos 4 algoritmos)')
axs[0].set_xticks(range(41)); axs[0].tick_params(axis='x', labelsize=7)
sns.heatmap(freq * 100, ax=axs[1], cmap='Blues', vmin=0, vmax=100, yticklabels=ordem, cbar_kws={'label': '% das execuções'})
axs[1].set_xlabel('índice do atributo'); axs[1].set_title('Por algoritmo')
plt.tight_layout(); salvar_fig(fig, 'rep_03_frequencia_atributos'); plt.show()
top = pd.Series(geral, index=range(41)).sort_values(ascending=False).head(8)
print('Atributos mais frequentes (índice: % das soluções):', {int(k): round(v) for k, v in top.items()})
print('Nº médio de atributos selecionados:', df.groupby('algo')['n_features'].mean().round(1).to_dict())
""")

md(nb, r"""
**Leitura.** Poucos atributos aparecem em quase todas as soluções e as buscas selecionam subconjuntos **bem menores que 41**. Note que a coluna **24** é o *instance weight* do UCI (peso amostral do censo), que **não deveria ser um atributo de entrada**; se ela aparece com frequência nas soluções, a árvore está se beneficiando de um atributo que, conceitualmente, não descreve a pessoa. O Notebook 2 inclui um experimento opcional para isso.

## 9. Divergências entre o artigo e o código

Mantivemos **o que o código faz**, pois a tarefa é replicar com o código oficial. As diferenças que identificamos:

| Item | O que diz o artigo | O que o `Example.ipynb` faz | Impacto esperado |
|---|---|---|---|
| Tamanho da amostra de subconjuntos `q` | `q = 20` | `q = 10` (`CHCqx(dataset, 10, 10, 2, 50)` — o 2º `10` é `n_individual`=q) | Meta-modelo calibrado com menos pontos → correlação de Spearman mais ruidosa, mas execução mais rápida |
| Orçamento de tempo do CHC | CHC roda pelo **mesmo tempo de CPU** que o CHC_QX | CHC para por convergência (`max_no_change=10`), sem orçamento de tempo | CHC puro leva muito mais tempo que o QX; a comparação deixa de ser "a tempo igual" |
| Embaralhamento | As instâncias são embaralhadas | `shuffle=False`: a divisão 60/20/20 segue a ordem do arquivo | Divisão determinística; se o arquivo tiver alguma ordem sistemática, treino/validação/teste podem diferir em distribuição (aqui a proporção de positivos é parecida) |

Também vale registrar: o próprio código tem outras particularidades que **não** alteramos — por exemplo, a variabilidade entre execuções vem apenas das sementes dos algoritmos (a divisão dos dados é fixa), e a busca usa a mesma validação para guiar e para escolher a melhor solução, o que favorece algum *overfitting* à validação.

## 10. Limitações observadas — por que a acurácia esconde o problema

Todos os algoritmos otimizam **acurácia**. Em uma base com 93,8 % de uma única classe, maximizar acurácia empurra a busca para modelos que **quase só preveem a classe majoritária**. Verificamos isso calculando, para as soluções replicadas, as métricas completas no teste:
""")

code(nb, r"""
cols = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_mcc', 'test_roc_auc', 'n_features']
resumo_metricas = tabela_med_dp(df, 'algo', cols,
                                escala={c: 1 for c in cols}, casas=3).loc[ordem]
resumo_metricas.columns = ['acurácia', 'acur. balanceada', 'precisão', 'recall (cl. 1)', 'F1 (cl. 1)', 'MCC', 'ROC-AUC', 'nº atributos']
display(resumo_metricas)
print('Referência — classe majoritária: acurácia 0.937 | acurácia balanceada 0.500 | recall 0.000 | F1 0.000')

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a, m in zip(ordem, ['o', 's', '^', 'D']):
    s = df[df.algo == a]
    axs[0].scatter(s.test_recall, 100 * s.test_accuracy, marker=m, s=45, label=a, alpha=.85)
axs[0].axhline(100 * (1 - ds.y_test.mean()), ls=':', c='#C0392B'); axs[0].set_xlabel('recall da classe positiva'); axs[0].set_ylabel('acurácia (%)')
axs[0].scatter(recall_score(ds.y_test, pred_dt), 100 * ds.TestAccuracy, marker='*', s=160, c='k', label='DT, todos os atributos')
axs[0].set_title('Acurácia alta, recall baixo'); axs[0].legend(fontsize=8, loc='center right'); axs[0].set_xlim(0, 1)
cm = np.array([[df.test_tn.median(), df.test_fp.median()], [df.test_fn.median(), df.test_tp.median()]])
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False, ax=axs[1], xticklabels=['prev. ≤50K', 'prev. >50K'], yticklabels=['real ≤50K', 'real >50K'])
axs[1].set_title('Matriz de confusão (mediana das execuções)')
plt.tight_layout(); salvar_fig(fig, 'rep_04_acuracia_vs_recall'); plt.show()
""")

md(nb, r"""
**Leitura (o ponto central para a melhoria).**

* As soluções "boas" (~94,9 % de acurácia) ficam apenas ~1,2 pp acima do classificador que sempre responde "≤ 50K" (≈ 93,7 %).
* O **recall da classe positiva é baixo** (≈ 0,24 nas soluções replicadas): o modelo só encontra uma em cada quatro pessoas de renda alta, embora acerte a maior parte do que prevê como positivo (precisão ≈ 0,83). A acurácia balanceada fica perto de 0,62 (0,5 = chute) e o F1 da classe 1 em torno de 0,37.
* O desbalanceamento não é tratado pelo artigo — os próprios autores reconhecem isso na seção 7 (*Limitations*): *"We have never considered the class imbalance case as it is not the focus of this paper. In all experiments accuracy was used to evaluate the fitness of feature subsets. This metric could naturally be replaced with one that accounts for the imbalance case, e.g., average recall."* Eles também sugerem amostragem estratificada das instâncias do meta-modelo.
* **A busca por acurácia piorou o recall em relação à árvore com todos os atributos.** Com os 41 atributos a árvore tem recall ≈ 0,47 e F1 ≈ 0,45 (tabela da seção 5); as soluções "otimizadas" têm recall ≈ 0,24 e F1 ≈ 0,37. Elas ganham acurácia porque quase não erram a classe negativa (precisão alta, ≈ 0,83), mas deixam de encontrar metade dos positivos que a árvore completa encontrava.
* **ROC-AUC:** com poucos atributos as folhas da árvore ficam impuras, então `predict_proba` carrega alguma informação de ranking (AUC ≈ 0,67–0,88 entre execuções, com medianas por algoritmo de 0,78 a 0,88, variando bastante entre soluções com acurácia praticamente idêntica). Mesmo assim, a AUC não aparece na fitness e não corrige o limiar de decisão.

Isso define o rumo do **Notebook 2**: manter dados, divisão, algoritmos e hiperparâmetros, e trocar apenas a **função de fitness** (e, em variantes, o classificador/treino) para uma que enxergue a classe de interesse.

## 11. Salvando os resultados
""")

code(nb, r"""
RES_DIR.mkdir(exist_ok=True)
saida = df.copy(); saida['features'] = saida['features'].apply(lambda f: ' '.join(map(str, f)))
saida.to_csv(RES_DIR / 'replicacao_runs.csv', index=False)

resumo_csv = pd.concat([
    tabela2.rename(columns={'Nossa replicação (mediana ± dp)': 'replicacao', 'Artigo (mediana ± dp)': 'artigo',
                            'Δ mediana (pp)': 'delta_pp', 'Dentro da tolerância?': 'dentro_tolerancia'}).set_index('Algoritmo'),
    resumo_metricas.reindex(tabela2['Algoritmo']).add_prefix('m_')], axis=1)
resumo_csv.to_csv(RES_DIR / 'replicacao_resumo.csv')
print('salvos: results/replicacao_runs.csv e results/replicacao_resumo.csv'); resumo_csv
""")

md(nb, r"""
## Resumo para o relatório

* **Descrição da base:** census-income (UCI Census-Income KDD) — 199.523 instâncias × 41 atributos numéricos já codificados + rótulo (renda > 50K), sem valores faltantes. **Desbalanceada: 93,79 % classe 0 e 6,21 % classe 1.** A coluna 24 é o *instance weight* (não deveria ser atributo de entrada; o código a usa como atributo).
* **Tamanho da população:** 50 indivíduos (CHC) / 50 partículas (PSO), nos quatro algoritmos. Meta-modelo QX: `q = 10` subconjuntos de referência, controle de evolução a cada `f = 10` gerações, `f_no_change = 2`.
* **% treino/validação/teste:** 60 % / 20 % / 20 %, **sequencial, sem embaralhar** (`shuffle=False`), normalização *z-score* ajustada só no treino. Fitness na validação; teste só na avaliação final.
* **Validação cruzada:** **não é usada.** O artigo usa *holdout* 60/20/20; a robustez estatística vem de **10 repetições independentes** (sementes 0–9) dos algoritmos estocásticos, reportando mediana ± desvio padrão. A divisão dos dados é fixa em todas as repetições.
* **Algoritmos:** CHC_QX, PSO_QX, CHC e PSO para seleção de atributos (*wrapper*), com Árvore de Decisão (`random_state=0`) como classificador.
* **Critério dos hiperparâmetros:** os valores do artigo / notebook oficial, **sem *Grid Search***. Divergências conhecidas entre artigo e código (q=20 vs 10; tempo de CPU do CHC; embaralhamento) foram documentadas e mantivemos o que o código faz.
* **Transfer learning, *dropout*, aumento de dados:** **não se aplicam** — não há rede neural nem pré-treinamento; a base é tabular e o modelo é uma árvore de decisão. (Aumento de dados só aparece no Notebook 2, como SMOTE para tratar o desbalanceamento.)
* **Desbalanceamento:** o original **não trata**; a acurácia esconde recall baixo da classe positiva (seção 10). A melhoria (Notebook 2) trata.
* **Métricas:** acurácia (a do artigo), acurácia balanceada, precisão, recall e F1 da classe 1, MCC, ROC-AUC, matriz de confusão, nº de atributos e tempo até a melhor solução.
""")


