"""Executor de experimentos: 1 execução = (configuração, algoritmo, semente).

Cada execução é salva em results/runs/<config>__<algo>__s<seed>.json assim que termina,
então o notebook pode ser interrompido e retomado sem refazer nada.
"""
import os
# Uma thread por processo: o paralelismo vem do joblib (e a comparação de tempo fica justa).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import sys
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

RAIZ = Path(__file__).resolve().parent.parent          # nossa_replica/
ORIGINAL = RAIZ.parent / 'projeto_original'            # código e dados dos autores
if not (ORIGINAL / 'code' / 'Evolution.py').exists():
    raise FileNotFoundError(f'{ORIGINAL} não encontrado. Rode antes, a partir de nossa_replica/:  python setup_original.py')
sys.path.insert(0, str(ORIGINAL / 'code'))
sys.path.insert(0, str(RAIZ / 'melhoria'))
from Dataset import Dataset            # noqa: E402
from Evolution import Evolution        # noqa: E402
from DatasetMod import DatasetMod      # noqa: E402

ARQUIVO_DADOS = str(ORIGINAL / 'data' / 'census-income.data')
RUNS_DIR = RAIZ / 'results' / 'runs'

ALGOS = ['CHC_QX', 'PSO_QX', 'CHC', 'PSO']

# Configurações. 'original' usa a classe Dataset INTACTA dos autores.
CONFIGS = {
    'original':    dict(mod=False),
    'M1_f1':       dict(mod=True, metric='f1'),
    'M2_bacc_bal': dict(mod=True, metric='balanced_accuracy', class_weight='balanced'),
    'M3_f1_smote': dict(mod=True, metric='f1', smote=True),
    # Ablações: separam o efeito de cada ingrediente de M2 e M3
    'M2b_bacc':      dict(mod=True, metric='balanced_accuracy'),            # M2 sem class_weight
    'M3b_acc_smote': dict(mod=True, metric='accuracy', smote=True),          # M3 com acurácia (sem F1)
    # Outra métrica balanceada
    'M4_mcc':        dict(mod=True, metric='mcc'),
    # Experimento extra: remove a coluna 24 (instance weight do UCI)
    'X1_acc_sem24': dict(mod=True, metric='accuracy', drop_cols=[24]),
    'X2_f1_sem24':  dict(mod=True, metric='f1', drop_cols=[24]),
}

# Hiperparâmetros EXATOS do Example.ipynb oficial
POP = 50
OPTIONS = {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298, 'k': POP, 'p': 2}


def carregar_dataset(config):
    """Carrega a base e faz a divisão 60/20/20 exatamente como no notebook oficial."""
    cfg = CONFIGS[config]
    clf = DecisionTreeClassifier(random_state=0, class_weight=cfg.get('class_weight'))
    if cfg['mod']:
        ds = DatasetMod(ARQUIVO_DADOS, ',', -1, divide_dataset=False, header=None,
                        metric=cfg['metric'], smote=cfg.get('smote', False),
                        drop_cols=cfg.get('drop_cols', []))
    else:
        ds = Dataset(ARQUIVO_DADOS, ',', -1, divide_dataset=False, header=None)
    ds.divide_dataset(clf, normalize=True, shuffle=False, all_features=True,
                      all_instances=True, evaluate=True, partial_sample=False)
    return ds


def colunas_originais(config):
    """Mapeia índice de atributo (após eventual drop) -> índice de coluna do arquivo."""
    drop = CONFIGS[config].get('drop_cols', [])
    return [c for c in range(41) if c not in drop]


def metricas(ds, feats, prefixo, X, y, clf):
    pred = clf.predict(X[:, feats])
    proba = clf.predict_proba(X[:, feats])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        f'{prefixo}_accuracy': accuracy_score(y, pred),
        f'{prefixo}_balanced_accuracy': balanced_accuracy_score(y, pred),
        f'{prefixo}_precision': precision_score(y, pred, zero_division=0),
        f'{prefixo}_recall': recall_score(y, pred, zero_division=0),
        f'{prefixo}_f1': f1_score(y, pred, zero_division=0),
        f'{prefixo}_mcc': matthews_corrcoef(y, pred),
        f'{prefixo}_roc_auc': roc_auc_score(y, proba),
        f'{prefixo}_tn': int(tn), f'{prefixo}_fp': int(fp), f'{prefixo}_fn': int(fn), f'{prefixo}_tp': int(tp),
    }


def executar_algoritmo(algo, ds):
    """Chama os algoritmos com os mesmos argumentos do Example.ipynb. Retorna (vetor 0/1, tempo)."""
    ind_size = ds.X_train.shape[1]
    if algo == 'CHC_QX':
        log, _ = Evolution.CHCqx(ds, 10, 10, 2, POP, verbose=0)
        sol, t = log.iloc[-1]['ind'], log.iloc[-1]['time']
    elif algo == 'PSO_QX':
        log, _ = Evolution.PSOqx(ds, OPTIONS, 10, 10, 2, POP, verbose=0)
        sol, t = log.iloc[-1]['ind'], log.iloc[-1]['time']
    elif algo == 'CHC':
        toolbox = Evolution.create_toolbox('feature_selection', 'validation', ds, None)
        pop = Evolution.create_population(POP, ind_size)
        d = ind_size // 4
        log, _, _ = Evolution.CHC(ds, toolbox, d, pop, verbose=0, max_no_change=10)
        sol, t = log.iloc[-1]['best_solution'], log.iloc[-1]['time']
    elif algo == 'PSO':
        log = Evolution.PSO(ds, OPTIONS, POP, steps_no_change=10, verbose=0)
        sol, t = log.iloc[-1]['ind'], log.iloc[-1]['time']
    else:
        raise ValueError(algo)
    return np.array(sol), float(t)


def caminho(config, algo, seed):
    return RUNS_DIR / f'{config}__{algo}__s{seed}.json'


def run_one(config, algo, seed, force=False):
    """Roda (ou recupera do disco) uma execução."""
    arq = caminho(config, algo, seed)
    if arq.exists() and not force:
        return json.loads(arq.read_text())
    warnings.filterwarnings('ignore')
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    ds = carregar_dataset(config)
    sol, t_melhor = executar_algoritmo(algo, ds)
    feats = list(np.where(sol == 1)[0])
    # Reavalia a solução final: treino completo -> validação e TESTE (teste só aqui, no fim)
    clf = clone(ds.clf)
    clf.fit(ds.X_train[:, feats], ds.y_train)
    cols = colunas_originais(config)
    res = dict(config=config, algo=algo, seed=seed,
               n_features=len(feats), features=[int(cols[f]) for f in feats],
               time_best=t_melhor, time_total=time.time() - t0)
    res.update(metricas(ds, feats, 'val', ds.X_val, ds.y_val, clf))
    res.update(metricas(ds, feats, 'test', ds.X_test, ds.y_test, clf))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix('.tmp')
    tmp.write_text(json.dumps(res))
    tmp.rename(arq)   # escrita atômica
    return res


# Custo relativo estimado (medido no teste com 1 semente): usado só para ordenar as tarefas
# das mais caras para as mais baratas, o que melhora o aproveitamento dos processos.
_CUSTO_CFG = {'M3_f1_smote': 4, 'M3b_acc_smote': 2, 'M1_f1': 3, 'M4_mcc': 3, 'X2_f1_sem24': 3}
_CUSTO_ALGO = {'PSO': 3, 'CHC': 2, 'PSO_QX': 1, 'CHC_QX': 1}


def run_many(configs, algos, seeds, n_jobs=4, force=False, executar=True):
    """Executa em paralelo só o que ainda não está salvo; retorna DataFrame com tudo.
    executar=False apenas lê o que já existe em disco (útil para só gerar tabelas/figuras)."""
    from joblib import Parallel, delayed
    jobs = [(c, a, s) for a in algos for c in configs for s in seeds if force or not caminho(c, a, s).exists()]
    jobs.sort(key=lambda j: -_CUSTO_CFG.get(j[0], 1) * _CUSTO_ALGO[j[1]])
    print(f'{len(jobs)} execuções pendentes (de {len(configs) * len(algos) * len(seeds)}).')
    if jobs and executar:
        Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(c, a, s, force) for c, a, s in jobs)
    return carregar_resultados(configs, algos, seeds)


def carregar_resultados(configs, algos, seeds):
    linhas = [json.loads(caminho(c, a, s).read_text())
              for c in configs for a in algos for s in seeds if caminho(c, a, s).exists()]
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- transferência para outros classificadores
TRANSF_DIR = RAIZ / 'results' / 'transfer'
_DS_BASE = {}


def _dataset_base():
    """Base original (41 atributos, sem SMOTE/pesos): igual para todos os subconjuntos."""
    if 'ds' not in _DS_BASE:
        _DS_BASE['ds'] = carregar_dataset('original')
    return _DS_BASE['ds']


def avaliar_transferencia(config, algo, seed, force=False):
    """Pega o subconjunto de atributos JÁ escolhido numa execução e o avalia com outros
    classificadores (Regressão Logística e Random Forest), sem fazer nenhuma busca nova.
    Treina no treino original (sem SMOTE) e mede no teste. Salva em results/transfer/."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    arq = TRANSF_DIR / f'{config}__{algo}__s{seed}.json'
    if arq.exists() and not force:
        return json.loads(arq.read_text())
    warnings.filterwarnings('ignore')
    run = json.loads(caminho(config, algo, seed).read_text())
    feats = run['features']                      # índices das colunas do arquivo original
    ds = _dataset_base()
    modelos = {'LR': LogisticRegression(max_iter=300, class_weight='balanced'),
               'RF': RandomForestClassifier(n_estimators=30, min_samples_leaf=5, class_weight='balanced_subsample',
                                            n_jobs=1, random_state=0)}
    res = dict(config=config, algo=algo, seed=seed, n_features=len(feats))
    for nome, m in modelos.items():
        m.fit(ds.X_train[:, feats], ds.y_train)
        proba = m.predict_proba(ds.X_test[:, feats])[:, 1]
        pred = (proba >= 0.5).astype(int)
        res[f'{nome}_roc_auc'] = roc_auc_score(ds.y_test, proba)
        res[f'{nome}_pr_auc'] = average_precision_score(ds.y_test, proba)
        res[f'{nome}_recall'] = recall_score(ds.y_test, pred, zero_division=0)
        res[f'{nome}_f1'] = f1_score(ds.y_test, pred, zero_division=0)
    TRANSF_DIR.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix('.tmp'); tmp.write_text(json.dumps(res)); tmp.rename(arq)
    return res


def transferencia_todas(runs_df, n_jobs=4, executar=True):
    """Roda avaliar_transferencia para todas as execuções de runs_df (em paralelo)."""
    from joblib import Parallel, delayed
    chaves = list(zip(runs_df.config, runs_df.algo, runs_df.seed))
    pend = [k for k in chaves if not (TRANSF_DIR / f'{k[0]}__{k[1]}__s{k[2]}.json').exists()]
    print(f'{len(pend)} avaliações de transferência pendentes (de {len(chaves)}).')
    if pend and executar:
        Parallel(n_jobs=n_jobs, verbose=5)(delayed(avaliar_transferencia)(*k) for k in pend)
    return pd.DataFrame([json.loads((TRANSF_DIR / f'{c}__{a}__s{s}.json').read_text())
                         for c, a, s in chaves if (TRANSF_DIR / f'{c}__{a}__s{s}.json').exists()])
