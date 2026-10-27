# Replicação e melhoria — Fast Genetic Algorithm for Feature Selection (census-income)

Miniprojeto de IA Avançada (UNICAP). Artigo: Altarabichi et al., *Expert Systems with Applications* 211 (2023) 118528 ([arXiv:2404.03996](https://arxiv.org/pdf/2404.03996)). Código oficial: <https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection>.

## Estrutura

```
projeto_original/                 # o projeto DOS AUTORES — NÃO versionado aqui; criado por `python setup_original.py`
                                  # (3 correções de compatibilidade aplicadas sobre o código oficial)
├── code/                         Dataset.py, Evolution.py, Example.ipynb
├── data/census-income.data
└── results/results.csv
nossa_replica/                    # o NOSSO trabalho
├── projeto_replica_e_melhoria.ipynb   # notebook único: Parte A (replicação), B (melhoria), C (resumo)
├── melhoria/DatasetMod.py        # a ÚNICA mudança de lógica da melhoria (subclasse de Dataset)
├── src/runner.py                 # executa 1 execução (config, algoritmo, semente) → results/runs/*.json
├── src/analise.py                # tabelas, testes pareados, paleta, salvar_fig
├── results/                      # runs/ (1 JSON por execução) + CSVs finais
├── figures/                      # PNG 150 dpi para slides
└── .build/                       # scripts que geram o .ipynb (opcional)
```

## Como rodar

**Local** (Python 3.10+), a partir da pasta `nossa_replica/`:
```bash
python setup_original.py          # 1ª vez: baixa o código/dados oficiais (commit 6ac5a7e) e aplica as 3 correções
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # ou: pip install deap pyswarms scikit-learn pandas numpy scipy matplotlib seaborn imbalanced-learn joblib jupyter
jupyter lab projeto_replica_e_melhoria.ipynb
```
**Modo rápido (testar o pipeline):** `N_RUNS=1 jupyter lab ...` (ou `export N_RUNS=1`). Variáveis opcionais: `N_JOBS`, `N_RUNS_CHC`, `RODAR_M3=0/1`, `RODAR_ABLACOES=0/1` (M2b, M3b, M4), `RODAR_EXTRA=0/1` (sem coluna 24), `EXECUTAR=0` (só lê o que já está em `results/runs/`), `EXECUTAR_TRANSF=0/1`.
Execução sem abrir o Jupyter: `jupyter nbconvert --to notebook --execute --inplace projeto_replica_e_melhoria.ipynb`.

**Colab:** envie a pasta `nossa_replica` para o Drive, monte o Drive, `%cd` para ela e rode `!python setup_original.py` e `!pip install deap pyswarms imbalanced-learn`. O Colab gratuito tem poucos núcleos: reduza `N_JOBS`/`N_RUNS`.

Sementes `0..N_RUNS-1`; `OMP_NUM_THREADS=1` é definido automaticamente; cada execução é salva ao terminar — se o kernel cair, reexecute e só o que falta é refeito. Para refazer do zero, apague `results/runs/`.

## Quanto tempo leva

Tempo total **mediano de uma execução** (minutos), medido nas 10 sementes, num Mac de 10 núcleos com 8 execuções em paralelo (serve como ordem de grandeza):

| Config | CHC_QX | PSO_QX | CHC | PSO |
|---|---|---|---|---|
| original (acurácia) | 1,2 | 1,2 | 3,9 | 4,5 |
| M1 (F1) | 3,5 | 4,7 | 14,4 | 14,4 |
| M2 (bal. acc + class_weight) | 1,6 | 1,3 | 5,3 | 8,0 |
| M2b / M4 / X2 | ≈ 4 | ≈ 4 | ≈ 14–19 | ≈ 15–19 |
| M3 (F1 + SMOTE) | 11,5 | 8,2 | 51,9 | **55,9** |
| M3b (acurácia + SMOTE) | 5,0 | 4,1 | 20,4 | 12,7 |
| X1 (acurácia sem col. 24) | 1,3 | 1,0 | 3,3 | 4,7 |

Pacote completo (9 configurações × 4 algoritmos × 10 sementes = 360 execuções): **≈ 7,6 h** (458 min) + ≈ 15–30 min para a transferência. Só a Parte A: ≈ 10–15 min. Em 2 núcleos, multiplique por ~4. F1 demora mais porque a busca explora subconjuntos grandes (~25–30 atributos); M3 (SMOTE) é ≈ 3,5× mais lento que M1.

## Resultados em uma olhada (10 sementes, teste)

* **Replicação:** medianas de acurácia CHC 94,92, CHC_QX 94,93, PSO 94,91, PSO_QX 94,95 (artigo: 94,63 / 94,94 / 94,79 / 94,94). Baseline DT: 92,815 (artigo: 92,87; causa da diferença não identificada).
* **Melhoria (M1, F1 como fitness):** acurácia ≈ 94,9 → ≈ 93,1; recall ≈ 0,24 → ≈ 0,48; F1 ≈ 0,376 → ≈ 0,466; MCC praticamente inalterado (≈ 0,43). Detalhes e testes estatísticos no notebook (B3–B8). Um documento de apoio para montar a apresentação está em `DOCUMENTO_BASE_PARA_ROTEIRO.md`.

## Reprodutibilidade

`requirements.txt` traz as versões usadas (Python 3.13). Para refazer tudo do zero, apague `results/runs/` e `results/transfer/`.
