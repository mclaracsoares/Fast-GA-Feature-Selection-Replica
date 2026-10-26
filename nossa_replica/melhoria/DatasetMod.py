"""Melhoria proposta: troca da fitness (e, opcionalmente, do treino) isolada numa subclasse.

Nada em code/Dataset.py nem em code/Evolution.py é alterado (além das correções de
compatibilidade). Como Evolution.py sempre chama set_validation_accuracy() /
get_validation_accuracy(), sobrescrever esses métodos muda a fitness dos 4 algoritmos
E o critério do meta-modelo do QX (correlação de Spearman passa a ser calculada
sobre a nova métrica).
"""
import sys
import os
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'projeto_original', 'code'))
from Dataset import Dataset  # noqa: E402

# zero_division=0: se a árvore não prevê nenhum positivo, F1 = 0 (é o que o sklearn já
# retornaria, só que sem emitir um aviso a cada avaliação).
METRICAS = {'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score,
            'f1': partial(f1_score, zero_division=0),
            'mcc': matthews_corrcoef}

# Cache do SMOTE: o treino é determinístico (shuffle=False), então o SMOTE só precisa
# rodar uma vez por processo.
_CACHE_SMOTE = {}


class DatasetMod(Dataset):
    def __init__(self, *args, metric='accuracy', smote=False, drop_cols=(), **kwargs):
        self.metric_fn = METRICAS[metric]
        self.smote = smote            # SMOTE só no treino, depois da divisão e da normalização
        self.drop_cols = list(drop_cols)  # colunas (rótulos originais) removidas antes da divisão
        super().__init__(*args, **kwargs)

    # ---- fitness: a única mudança "obrigatória" da melhoria ------------------------
    def set_validation_accuracy(self):
        y_pred = self.clf.predict(self.X_val[:, self.features])
        self.ValidationAccuracy = self.metric_fn(self.y_val, y_pred)

    def set_test_accuracy(self):
        y_pred = self.clf.predict(self.X_test[:, self.features])
        self.TestAccuracy = self.metric_fn(self.y_test, y_pred)

    # ---- extras opcionais (M3 e experimento da coluna 24) -------------------------
    def divide_dataset(self, classifier, normalize=True, shuffle=True, all_features=True,
                       all_instances=True, evaluate=True, partial_sample=False):
        if self.drop_cols:
            # drop por rótulo é idempotente (divide_dataset é chamado várias vezes em cópias)
            self.df = self.df.drop(columns=self.drop_cols, errors='ignore')
            self.df_sampled = self.df

        if not self.smote:
            return super().divide_dataset(classifier, normalize, shuffle, all_features,
                                          all_instances, evaluate, partial_sample)

        # 1) divisão + normalização originais, sem avaliar e sem amostrar instâncias
        super().divide_dataset(classifier, normalize, shuffle, all_features,
                               True, False, False)
        # 2) SMOTE somente no treino (validação e teste ficam intactos)
        chave = (self.X_train.shape, int(self.y_train.sum()))
        if chave not in _CACHE_SMOTE:
            from imblearn.over_sampling import SMOTE
            _CACHE_SMOTE[chave] = SMOTE(random_state=0).fit_resample(self.X_train, self.y_train)
        self.X_train, self.y_train = _CACHE_SMOTE[chave]
        # 3) refaz a escolha de instâncias sobre o treino já reamostrado
        n = self.X_train.shape[0]
        if all_instances:
            self.instances = list(range(n))
        else:
            self.instances = list(np.where(np.random.choice([0, 1], size=(n,), p=[0.5, 0.5]) == 1)[0])
        if partial_sample:
            self.instances = np.random.choice(n, partial_sample, replace=False)
        if evaluate:
            self.fit_classifier()
            self.set_validation_accuracy()
            self.set_test_accuracy()
