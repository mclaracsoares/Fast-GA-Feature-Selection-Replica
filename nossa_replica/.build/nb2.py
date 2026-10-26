from nbhelp import *
nb = novo()

md(nb, r"""
# Notebook 2 — Melhoria: otimizar uma fitness que enxergue a classe minoritária

**Mesma base (census-income), mesma divisão 60/20/20, mesmos 4 algoritmos, mesmos hiperparâmetros** do Notebook 1. A única coisa que muda é *o que a busca otimiza*.

## 1. Motivação

No Notebook 1 vimos que:

* a base é **desbalanceada** (≈ 6,2 % de positivos, renda > 50K);
* um classificador que **sempre responde "≤ 50K"** já tem **≈ 93,7 %** de acurácia no teste — **mais** que a Árvore de Decisão com todos os atributos (92,87 % no artigo);
* as soluções "boas" encontradas pelo código (~94,9 %) ficam só ~1,2 pp acima disso, e com **recall da classe positiva ≈ 0,24**: encontram uma em cada quatro pessoas de renda alta, acurácia balanceada ≈ 0,62 e F1 ≈ 0,37.

Ou seja: otimizar **acurácia** recompensa modelos que ignoram a classe de interesse. Os próprios autores reconhecem a limitação na seção 7 do artigo (*Limitations*):

> *"We have never considered the class imbalance case as it is not the focus of this paper. In all experiments accuracy was used to evaluate the fitness of feature subsets. This metric could naturally be replaced with one that accounts for the imbalance case, e.g., average recall."*

Além da métrica, existe uma observação adicional que o artigo não considera: pela documentação do UCI, a **coluna 24 é o *instance weight*** (peso amostral do censo), que **não deveria ser usada como atributo de entrada**.

## 2. A modificação — isolada numa subclasse

`Evolution.py` **sempre** chama `set_validation_accuracy()` e `get_validation_accuracy()` para obter a fitness. Logo, basta uma subclasse de `Dataset` que sobrescreva esses métodos para mudar:

1. a **fitness dos 4 algoritmos** (CHC, PSO, CHC_QX, PSO_QX); **e**
2. o **critério do meta-modelo do QX**: `select_instances` e `evaluate_selected_instances` também leem `get_validation_accuracy()`, então a correlação de Spearman passa a ser calculada **sob a nova métrica**. Isto é coerente com o argumento do artigo (o meta-modelo deve concordar em *ranking* com a função original — agora a original é F1 ou acurácia balanceada).

**Nenhuma linha** de `Dataset.py` ou `Evolution.py` é tocada — a melhoria vive em `melhoria/DatasetMod.py`:
""")

code(nb, r"""
import os, sys, warnings
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
warnings.filterwarnings('ignore')

from pathlib import Path
RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(RAIZ / 'src')); sys.path.insert(0, str(RAIZ / 'code')); sys.path.insert(0, str(RAIZ / 'melhoria'))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import runner
from analise import *

# ------------------------- PARÂMETROS -------------------------
N_RUNS = int(os.environ.get('N_RUNS', 10))   # repetições de PSO e PSO_QX (foco da disciplina). N_RUNS=1 = teste rápido
N_RUNS_CHC = int(os.environ.get('N_RUNS_CHC', N_RUNS))   # repetições de CHC e CHC_QX (reduza, ex. 3, se o tempo for curto — e diga isso no relatório)
RODAR_M3 = True      # M3 (SMOTE) é opcional
RODAR_EXTRA = False  # experimento extra: remover a coluna 24 (instance weight)
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))
# --------------------------------------------------------------
SEEDS_PSO, SEEDS_CHC = list(range(N_RUNS)), list(range(N_RUNS_CHC))
print(open(RAIZ / 'melhoria' / 'DatasetMod.py').read())
""")

md(nb, r"""
### Configurações comparadas (todas contra a original)

| Config | Fitness (validação) | Árvore | Ideia |
|---|---|---|---|
| `original` | acurácia | `DecisionTreeClassifier(random_state=0)` | O artigo, sem mudanças (usa a classe `Dataset` intacta) |
| `M1_f1` | **F1 da classe 1** | padrão | Troca só a métrica: média harmônica de precisão e recall da classe de interesse |
| `M2_bacc_bal` | **acurácia balanceada** (= recall médio, a sugestão dos autores) | `class_weight='balanced'` | Métrica + classificador sensível ao custo: erros na classe rara pesam ~15× mais |
| `M3_f1_smote` | F1 | padrão, treinada com **SMOTE** | Aumento de dados sintético só no treino |
| `M2b_bacc` *(ablação)* | acurácia balanceada | padrão (**sem** pesos) | Isola o efeito da métrica do efeito do `class_weight` em M2 |
| `M3b_acc_smote` *(ablação)* | **acurácia** | padrão, com **SMOTE** | Isola o efeito do SMOTE do efeito do F1 em M3 |
| `M4_mcc` *(métrica extra)* | **MCC** (Matthews) | padrão | Outra métrica balanceada, resumo único da matriz de confusão |
| `X1_acc_sem24` / `X2_f1_sem24` *(extra)* | acurácia / F1 | padrão, **sem a coluna 24** | Remove o *instance weight* (peso amostral do censo) dos atributos |

**Por que essas escolhas?**

* **F1 (M1)** foca na classe positiva e não "premia" o acerto fácil da classe majoritária. Se o modelo nunca prevê positivos, F1 = 0.
* **Acurácia balanceada (M2)** é exatamente o *recall médio* citado pelos autores. Combinamos com `class_weight='balanced'` porque o peso nas classes empurra o modelo a arriscar mais previsões positivas.
* **SMOTE (M3)** cria exemplos sintéticos da classe rara (interpolando vizinhos). É aplicado **somente no treino, depois da divisão 60/20/20 e da normalização** — validação e teste permanecem com a distribuição real, senão a avaliação seria artificial (vazamento). Ressalva: os atributos são códigos de categorias; interpolar entre códigos gera valores intermediários sem significado categórico, então o SMOTE aqui é uma aproximação.

**Por que as ablações?** M2 muda *duas* coisas ao mesmo tempo (métrica e pesos) e M3 também (métrica e SMOTE). Sem separar, não dá para saber qual ingrediente produz o efeito. `M2b` (métrica sem pesos) e `M3b` (SMOTE sem trocar a métrica) são os controles que respondem a isso.

**Regra de ouro mantida:** o conjunto de teste **nunca** é usado para escolher configuração ou hiperparâmetro. A fitness é sempre a métrica na validação; o teste só aparece na avaliação final de cada execução.

**Comparabilidade.** A configuração `original` reaproveita as mesmas execuções do Notebook 1 (mesmos arquivos em `results/runs/`, mesmas sementes). Em cada semente, a configuração modificada começa **da mesma população inicial** que a original, o que permite testes **pareados por semente**.
""")

code(nb, r"""
configs = (['original', 'M1_f1', 'M2_bacc_bal'] + (['M3_f1_smote'] if RODAR_M3 else [])
           + (['M2b_bacc', 'M3b_acc_smote', 'M4_mcc'] if RODAR_ABLACOES else [])
           + (['X1_acc_sem24', 'X2_f1_sem24'] if RODAR_EXTRA else []))
import time; t0 = time.time()
partes = [runner.run_many(configs, ['PSO', 'PSO_QX'], SEEDS_PSO, n_jobs=N_JOBS, executar=EXECUTAR),
          runner.run_many(configs, ['CHC', 'CHC_QX'], SEEDS_CHC, n_jobs=N_JOBS, executar=EXECUTAR)]
df = pd.concat(partes, ignore_index=True)
print(f'Tempo desta célula: {(time.time()-t0)/60:.1f} min | execuções: {len(df)}')
print(df.groupby(['algo', 'config']).size().unstack())
""")

md(nb, r"""
## 3. Resultados: tabela comparativa (mediana ± desvio padrão no teste)
""")

code(nb, r"""
ALGS = ['PSO_QX', 'PSO', 'CHC_QX', 'CHC']
CFGS = [c for c in configs if c in df.config.unique()]
cols = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_mcc', 'test_roc_auc', 'n_features', 'time_best']
nomes = ['acurácia (%)', 'acur. balanceada', 'precisão', 'recall', 'F1', 'MCC', 'ROC-AUC', 'nº atributos', 'tempo (s)']
esc = {c: 1 for c in cols}; esc['test_accuracy'] = 100

blocos = {}
for a in ALGS:
    sub = df[df.algo == a]
    t = tabela_med_dp(sub, 'config', cols, escala=esc, casas=3).reindex(CFGS)
    t.columns = nomes
    t['acurácia (%)'] = [med_dp(sub[sub.config == c].test_accuracy, 100, 2) for c in CFGS]
    t['nº atributos'] = [med_dp(sub[sub.config == c].n_features, 1, 1) for c in CFGS]
    t['tempo (s)'] = [med_dp(sub[sub.config == c].time_best, 1, 0) for c in CFGS]
    blocos[a] = t
    print(f'\n=== {a}  (n = {sub.groupby("config").size().to_dict()}) ===')
    display(t)
resumo = pd.concat(blocos, names=['algoritmo', 'config'])
""")

md(nb, r"""
**Como ler.** Em cada algoritmo, comparamos a linha `original` com as modificadas. A **acurácia** tende a *cair* (o modelo passa a arriscar mais previsões positivas, errando mais negativos), enquanto **recall**, **F1** e **acurácia balanceada** tendem a subir (o **MCC**, que pondera precisão e recall, fica praticamente igual — ver B8). A **precisão** pode cair — é o outro lado do mesmo trade-off.

## 4. Testes estatísticos: original × cada modificação (e comparações das ablações)

Teste **pareado por semente**: Wilcoxon (não paramétrico, indicado para n pequeno) e *t* pareado (o que o artigo utiliza). Com n = 10, o menor p-valor possível do Wilcoxon é ≈ 0,002. Com menos de 5 pares os testes não são computados (ficam `NaN`).

Além de `original` × cada modificação, comparamos pares que **isolam um ingrediente**:

| Comparação | O que isola |
|---|---|
| `M2b_bacc` × `M2_bacc_bal` | efeito do `class_weight` (mesma métrica) |
| `M1_f1` × `M3_f1_smote` | efeito do SMOTE sob F1 |
| `original` × `M3b_acc_smote` | efeito do SMOTE sob acurácia |
| `M1_f1` × `M4_mcc` | F1 versus MCC como fitness |
| `original` × `X1_acc_sem24` e `M1_f1` × `X2_f1_sem24` | efeito de remover a coluna 24 |
""")

code(nb, r"""
met = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_mcc']
pares = [('original', c) for c in CFGS if c != 'original']
pares += [p for p in [('M2b_bacc', 'M2_bacc_bal'), ('M1_f1', 'M3_f1_smote'), ('M1_f1', 'M4_mcc'), ('M1_f1', 'X2_f1_sem24')]
          if p[0] in CFGS and p[1] in CFGS]
linhas = []
for a in ALGS:
    for base, c in pares:
        t = testes_pareados(df, a, base, c, met); t.insert(0, 'config', c); t.insert(0, 'base', base); t.insert(0, 'algoritmo', a)
        linhas.append(t.reset_index(names='métrica'))
testes = pd.concat(linhas, ignore_index=True)
testes['sig (Wilcoxon, p<0,05)'] = np.where(testes.p_wilcoxon < .05, '*', '')
display(testes[testes['métrica'].isin(['test_f1', 'test_recall', 'test_accuracy'])].round(4))   # resumo; a tabela completa vai para o CSV
""")

md(nb, r"""
## 5. Gráficos

Cada configuração tem **uma cor fixa** em todas as figuras (paleta Okabe-Ito, segura para daltonismo). Todas as figuras são salvas em `figures/` (PNG, 150 dpi).

### 5.1 Precisão × recall de cada execução

Cada ponto é **uma execução**. A mudança de fitness desloca a nuvem de pontos para a direita (mais recall) e, em geral, para baixo (menos precisão): o trade-off em ação.
""")

code(nb, r"""
def cor(c): return CORES[c]
fig, axs = plt.subplots(1, 4, figsize=(16, 3.9), sharex=True, sharey=True)
for ax, a in zip(axs, ALGS):
    for c in CFGS:
        s = df[(df.algo == a) & (df.config == c)]
        ax.scatter(s.test_recall, s.test_precision, s=50, color=cor(c), edgecolor='white', linewidth=1.2, label=ROTULOS[c], alpha=.9)
    ax.set_title(a); ax.set_xlabel('recall (classe 1)'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel('precisão (classe 1)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.14))
plt.tight_layout(); salvar_fig(fig, 'mel_01_precisao_vs_recall'); plt.show()
""")

md(nb, r"""
### 5.2 Boxplots de F1, acurácia balanceada e recall
""")

code(nb, r"""
fig, axs = plt.subplots(3, 4, figsize=(16, 9), sharey='row')
for i, (m, nome) in enumerate([('test_f1', 'F1 (classe 1)'), ('test_balanced_accuracy', 'acurácia balanceada'), ('test_recall', 'recall (classe 1)')]):
    for j, a in enumerate(ALGS):
        sub = df[df.algo == a]
        sns.boxplot(data=sub, x='config', y=m, order=CFGS, palette=[cor(c) for c in CFGS], hue='config', hue_order=CFGS, legend=False, ax=axs[i, j], width=.6, fliersize=0)
        sns.stripplot(data=sub, x='config', y=m, order=CFGS, color='#222', size=3, ax=axs[i, j])
        axs[i, j].set_xlabel(''); axs[i, j].set_ylabel(nome if j == 0 else ''); axs[i, j].set_xticks(range(len(CFGS))); axs[i, j].set_xticklabels([c.split('_')[0] for c in CFGS])
        if i == 0: axs[i, j].set_title(a)
plt.tight_layout(); salvar_fig(fig, 'mel_02_boxplots_metricas'); plt.show()
""")

md(nb, r"""
### 5.3 Matrizes de confusão representativas

Para cada configuração mostramos a execução cujo **F1 é o mediano** daquele grupo (a "execução típica"), no conjunto de teste (≈ 39,9 mil instâncias; escolha automática, sem olhar qual é a melhor). Mostramos o PSO_QX (algoritmo-foco); a figura com os quatro algoritmos também é salva em `figures/`. Cada célula traz a contagem e a % da linha (classe real).
""")

code(nb, r"""
def run_mediana(sub, m='test_f1'):
    sub = sub.sort_values(m).reset_index(drop=True)
    return sub.iloc[len(sub) // 2]

def plot_cm(a, ax_row):
    for ax, c in zip(ax_row, CFGS):
        r = run_mediana(df[(df.algo == a) & (df.config == c)])
        cm = np.array([[r.test_tn, r.test_fp], [r.test_fn, r.test_tp]]); pct = cm / cm.sum(1, keepdims=True) * 100
        sns.heatmap(pct, annot=np.array([[f'{cm[i,j]}\n({pct[i,j]:.0f}%)' for j in range(2)] for i in range(2)]), fmt='', cmap='Blues', vmin=0, vmax=100,
                    cbar=False, ax=ax, xticklabels=['prev. ≤50K', 'prev. >50K'], yticklabels=['real ≤50K', 'real >50K'])
        ax.set_title(f'{ROTULOS[c]}\nrec={r.test_recall:.2f} prec={r.test_precision:.2f}', fontsize=8.5, color=cor(c))
        ax.set_ylabel(a if ax is ax_row[0] else '')
    return None

def grade_cm(a, nome):
    ncol = min(5, len(CFGS)); nlin = -(-len(CFGS) // ncol)
    fig, axs = plt.subplots(nlin, ncol, figsize=(3.3 * ncol, 3.5 * nlin), squeeze=False)
    plot_cm(a, axs.ravel()[:len(CFGS)])
    for ax in axs.ravel()[len(CFGS):]: ax.axis('off')
    plt.tight_layout(); salvar_fig(fig, nome); return fig
grade_cm('PSO_QX', 'mel_03_matrizes_confusao_PSO_QX'); plt.show()
for a in ['PSO', 'CHC_QX', 'CHC']: grade_cm(a, f'mel_03b_matrizes_confusao_{a}'); plt.close('all')
""")

md(nb, r"""
### 5.4 Número de atributos selecionados e tempo
""")

code(nb, r"""
fig, axs = plt.subplots(2, 4, figsize=(16, 6.4))
for j, a in enumerate(ALGS):
    sub = df[df.algo == a]
    for i, (m, nome, log) in enumerate([('n_features', 'nº de atributos selecionados', False), ('time_best', 'tempo até a melhor solução (s)', True)]):
        sns.boxplot(data=sub, x='config', y=m, order=CFGS, palette=[cor(c) for c in CFGS], hue='config', hue_order=CFGS, legend=False, ax=axs[i, j], width=.6, fliersize=0)
        sns.stripplot(data=sub, x='config', y=m, order=CFGS, color='#222', size=3, ax=axs[i, j])
        axs[i, j].set_xlabel(''); axs[i, j].set_ylabel(nome if j == 0 else ''); axs[i, j].set_xticks(range(len(CFGS))); axs[i, j].set_xticklabels([c.split('_')[0] for c in CFGS])
        if log: axs[i, j].set_yscale('log')
        if i == 0: axs[i, j].set_title(a)
plt.tight_layout(); salvar_fig(fig, 'mel_04_atributos_e_tempo'); plt.show()
""")

md(nb, r"""
### 5.5 Quais atributos cada configuração prefere?

Frequência de seleção dos atributos nas soluções finais (todas as execuções de todos os algoritmos de cada configuração). Se a coluna 24 (*instance weight*) aparece muito, o modelo está explorando um atributo que conceitualmente não deveria entrar.
""")

code(nb, r"""
freq = pd.DataFrame(0.0, index=CFGS, columns=range(41))
for c in CFGS:
    sub = df[df.config == c]
    for fs in sub.features: freq.loc[c, fs] += 100 / len(sub)
fig, ax = plt.subplots(figsize=(12, 2.2 + .35 * len(CFGS)))
sns.heatmap(freq, cmap='Blues', vmin=0, vmax=100, ax=ax, cbar_kws={'label': '% das soluções'}, yticklabels=[ROTULOS[c] for c in CFGS])
ax.set_xlabel('índice do atributo'); ax.set_title('Frequência de seleção de cada atributo')
plt.tight_layout(); salvar_fig(fig, 'mel_05_frequencia_atributos'); plt.show()
print('% de soluções que usam a coluna 24 (instance weight):', freq[24].round(0).to_dict())

# Estabilidade: similaridade de Jaccard média entre os subconjuntos de sementes diferentes (1 = sempre os mesmos atributos)
from itertools import combinations
def jaccard_medio(listas):
    sets = [set(l) for l in listas]
    v = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return np.mean(v) if v else np.nan
estab = pd.DataFrame({a: {c: jaccard_medio(df[(df.algo == a) & (df.config == c)].features) for c in CFGS} for a in ALGS})
print('\nEstabilidade dos subconjuntos entre sementes (Jaccard médio; 1 = idênticos):')
display(estab.round(2))
""")

md(nb, r"""
## 8. Discussão: o trade-off e a recomendação

Os números abaixo são calculados automaticamente a partir dos resultados (mediana das execuções, no teste).
""")

code(nb, r"""
def med(a, c, m): return df[(df.algo == a) & (df.config == c)][m].median()
for a in ALGS:
    print(f'--- {a} ---')
    for c in [c for c in CFGS if c != 'original']:
        print(f'  {c:13s} acurácia {100*med(a,"original","test_accuracy"):.2f}% → {100*med(a,c,"test_accuracy"):.2f}% | '
              f'recall {med(a,"original","test_recall"):.2f} → {med(a,c,"test_recall"):.2f} | '
              f'precisão {med(a,"original","test_precision"):.2f} → {med(a,c,"test_precision"):.2f} | '
              f'F1 {med(a,"original","test_f1"):.3f} → {med(a,c,"test_f1"):.3f} | '
              f'acc.bal. {med(a,"original","test_balanced_accuracy"):.3f} → {med(a,c,"test_balanced_accuracy"):.3f}')
print('\nReferência: classe majoritária = 93,70% de acurácia, recall 0, acurácia balanceada 0,50.')
""")

md(nb, r"""
**DISCUSSAO_PLACEHOLDER**
""")

code(nb, r"""
RES_DIR.mkdir(exist_ok=True)
saida = df.copy(); saida['features'] = saida['features'].apply(lambda f: ' '.join(map(str, f)))
saida.to_csv(RES_DIR / 'melhoria_runs.csv', index=False)
resumo.reset_index().to_csv(RES_DIR / 'melhoria_resumo.csv', index=False)
testes.to_csv(RES_DIR / 'melhoria_testes_estatisticos.csv', index=False)
print('salvos: results/melhoria_runs.csv, melhoria_resumo.csv, melhoria_testes_estatisticos.csv | figuras em figures/')
""")

