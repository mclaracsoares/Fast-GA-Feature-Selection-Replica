"""Funções de análise/visualização compartilhadas pelos dois notebooks."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
FIG_DIR = RAIZ / 'figures'
RES_DIR = RAIZ / 'results'
FIG_DIR.mkdir(exist_ok=True)

# Valores reportados no artigo (Tabela 2, census-income): acurácia de teste em %
ARTIGO = {'Baseline DT': (92.87, np.nan),
          'CHC': (94.63, 0.35), 'CHC_QX': (94.94, 0.07),
          'PSO': (94.79, 0.24), 'PSO_QX': (94.94, 0.06)}

NOMES = {'CHC_QX': 'CHC_QX', 'PSO_QX': 'PSO_QX', 'CHC': 'CHC', 'PSO': 'PSO'}

# Paleta Okabe-Ito (segura para daltonismo); a cor segue a configuração, nunca o ranking
CORES = {'original': '#0072B2', 'M1_f1': '#E69F00', 'M2_bacc_bal': '#009E73',
         'M3_f1_smote': '#CC79A7', 'X1_acc_sem24': '#56B4E9', 'X2_f1_sem24': '#8B5A2B',
         'M2b_bacc': '#332288', 'M3b_acc_smote': '#AA4499', 'M4_mcc': '#882255'}
ROTULOS = {'original': 'original (acurácia)', 'M1_f1': 'M1: F1',
           'M2_bacc_bal': 'M2: bal. acc + class_weight', 'M3_f1_smote': 'M3: F1 + SMOTE',
           'X1_acc_sem24': 'X1: acurácia sem col. 24', 'X2_f1_sem24': 'X2: F1 sem col. 24',
           'M2b_bacc': 'M2b: bal. acc (sem pesos)', 'M3b_acc_smote': 'M3b: acurácia + SMOTE', 'M4_mcc': 'M4: MCC'}

TEXTO = '#2b2b2b'
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.edgecolor': '#888', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'legend.frameon': False})


def salvar_fig(fig, nome):
    """PNG a 150 dpi em figures/, pronto para slides."""
    fig.savefig(FIG_DIR / f'{nome}.png', dpi=150, bbox_inches='tight', facecolor='white')


def med_dp(s, escala=100, casas=2):
    """'mediana ± desvio padrão' (ddof=1). Com 1 execução, o desvio fica em branco."""
    s = pd.Series(s).dropna() * escala
    if len(s) == 0:
        return '-'
    dp = s.std(ddof=1) if len(s) > 1 else np.nan
    return f'{s.median():.{casas}f} ± {dp:.{casas}f}' if len(s) > 1 else f'{s.median():.{casas}f} (n=1)'


def tabela_med_dp(df, por, cols, escala=None, casas=2):
    """Tabela agrupada com 'mediana ± dp' para cada coluna em cols."""
    escala = escala or {}
    linhas = {}
    for chave, g in df.groupby(por, sort=False):
        linhas[chave] = {c: med_dp(g[c], escala.get(c, 100 if c.startswith(('test_', 'val_')) and c not in
                                                  ('test_mcc', 'test_roc_auc') else 1),
                                   casas) for c in cols}
    return pd.DataFrame(linhas).T


def testes_pareados(df, algo, base, outra, metricas):
    """Wilcoxon pareado e t pareado por semente entre duas configurações (mesma semente = par)."""
    a = df[(df.algo == algo) & (df.config == base)].set_index('seed')
    b = df[(df.algo == algo) & (df.config == outra)].set_index('seed')
    sementes = a.index.intersection(b.index)
    saida = {}
    for m in metricas:
        x, y = a.loc[sementes, m].values, b.loc[sementes, m].values
        p_w = p_t = np.nan
        if len(sementes) >= 5:
            try:
                p_w = stats.wilcoxon(x, y).pvalue
            except ValueError:   # diferenças todas nulas
                p_w = 1.0
            p_t = stats.ttest_rel(x, y).pvalue
        saida[m] = dict(n_pares=len(sementes), mediana_base=np.median(x), mediana_mod=np.median(y),
                        delta_mediana=np.median(y) - np.median(x), p_wilcoxon=p_w, p_ttest=p_t)
    return pd.DataFrame(saida).T
