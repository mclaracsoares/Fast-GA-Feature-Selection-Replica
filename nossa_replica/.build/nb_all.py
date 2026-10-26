"""Monta o notebook único a partir das células de nb1.py (Parte A) e nb2.py (Parte B)."""
import re, copy
from nbhelp import *
import nb1, nb2

A, B = list(nb1.nb.cells), list(nb2.nb.cells)
nb = novo()

# ---------------------------------------------------------------- callouts "o que faz / por quê"
CALL = {
 'A': {
  '1': ("Apresenta o problema (seleção de atributos), a ideia do artigo (aproximação qualitativa, QX) e a lógica geral.",
        "Sem entender *o que* o artigo propõe, não dá para julgar se a replicação foi fiel nem onde a melhoria se encaixa."),
  '2': ("Lista os quatro algoritmos e os hiperparâmetros exatos do notebook oficial.",
        "Replicar \"exatamente\" exige fixar cada valor; qualquer diferença contaminaria a comparação com o artigo."),
  '3': ("Importa bibliotecas, fixa threads e sementes, define `N_RUNS`/`N_JOBS` e documenta as 3 correções de compatibilidade.",
        "Reprodutibilidade (sementes), comparação justa de tempo (1 thread por processo) e transparência sobre o que foi mexido no código dos autores."),
  '4': ("Mede dimensões, valores faltantes, distribuição de classes e a proporção de positivos em treino/validação/teste.",
        "Conhecer a base antes de modelar. O **desbalanceamento** descoberto aqui é a base de toda a Parte B."),
  '5': ("Calcula a Árvore de Decisão com todos os atributos e o classificador de classe majoritária no teste.",
        "A DT é o *Baseline* da Tabela 2 (checagem de que o pipeline bate com o artigo); o majoritário é a \"régua de sanidade\" que revela quanto a acurácia vale nesta base."),
  '6': ("Executa CHC_QX, PSO_QX, CHC e PSO `N_RUNS` vezes cada, em paralelo, gravando cada execução em `results/runs/`.",
        "Algoritmos estocásticos precisam de várias repetições (o artigo usa 10); a gravação incremental permite retomar se o kernel cair."),
  '7': ("Monta a tabela no formato da Tabela 2 (mediana ± desvio padrão), lado a lado com o artigo e com a diferença.",
        "É o critério objetivo de sucesso da replicação."),
  '8': ("Plota a acurácia por algoritmo, o tempo até a melhor solução e a frequência de seleção de cada atributo.",
        "Mostra visualmente a qualidade (acurácia), a contribuição central do artigo (tempo) e *quais* atributos importam."),
  '9': ("Registra onde o artigo e o código oficial divergem e o que mantivemos.",
        "Honestidade metodológica: a tarefa é replicar o código, então documentamos as diferenças em vez de \"corrigi-las\"."),
  '10': ("Calcula métricas completas das soluções replicadas (recall, F1, MCC, AUC…) e as compara com a árvore completa e com o classificador majoritário.",
         "Mostra que a acurácia alta **esconde** recall baixo — o problema que a Parte B vai atacar."),
  '11': ("Grava `results/replicacao_runs.csv` e `results/replicacao_resumo.csv`.",
         "Deixa os números prontos para o relatório e os slides."),
 },
 'B': {
  '1': ("Resume o problema encontrado na Parte A e cita o trecho do artigo que reconhece a limitação.",
        "Toda melhoria precisa de uma motivação mensurável; aqui ela vem dos próprios dados e dos autores."),
  '2': ("Implementa a mudança numa subclasse (`DatasetMod`) e define as configurações M1, M2, M3.",
        "Isolar a mudança garante que *só* a fitness muda: mesma base, divisão, algoritmos e hiperparâmetros."),
  '3': ("Tabela com todas as métricas (mediana ± dp) por algoritmo × configuração.",
        "Permite ver, lado a lado, o que se ganha e o que se perde com cada configuração."),
  '4': ("Teste de Wilcoxon e *t* pareado entre `original` e cada modificação, pareando por semente.",
        "Diferenças podem ser acaso; o teste pareado verifica se são consistentes entre as repetições."),
  '5': ("Gráficos: precisão×recall, boxplots, matrizes de confusão, nº de atributos, tempo e atributos preferidos.",
        "Cada um responde a uma pergunta (ver título de cada subseção); todos são salvos em `figures/` para os slides."),
  '6': ("Compara as ablações (M2b, M3b), a métrica MCC (M4) e o experimento sem a coluna 24 (X1, X2), com tabelas e testes pareados.",
        "Separar cada ingrediente da melhoria: sem isso não dá para saber *o que* produz o efeito (métrica, pesos, SMOTE ou a coluna 24)."),
  '7': ("Pega os subconjuntos de atributos já escolhidos (com árvore) e os avalia com Regressão Logística e Random Forest, sem nova busca.",
        "Verifica se a seleção \"generaliza\": se os atributos escolhidos pela fitness nova também são melhores para outros modelos, a melhoria é sobre os *atributos*, não um artefato da árvore."),
  '8': ("Interpreta o trade-off e recomenda quando usar cada configuração.",
        "É a conclusão do trabalho: números sem interpretação não sustentam uma recomendação."),
 }}

def decorar(cells, parte):
    for c in cells:
        if c.cell_type != 'markdown':
            continue
        t = c.source
        def h2(m):
            n, resto = m.group(1), m.group(2)
            out = f'## {parte}{n}. {resto}'
            if n in CALL[parte]:
                faz, pq = CALL[parte][n]
                out += f'\n\n> 🎯 **O que esta seção faz:** {faz}\n>\n> 💡 **Por que existe:** {pq}'
            return out
        t = re.sub(r'^## (\d+)\. (.*)$', h2, t, flags=re.M)
        t = re.sub(r'^### (\d+\.\d+) ', lambda m: f'### {parte}{m.group(1)} ', t, flags=re.M)
        t = (t.replace('Notebook 1', 'Parte A').replace('Notebook 2', 'Parte B')
              .replace('(seção 9)', '(seção A9)').replace('seção 10', 'seção A10').replace('da seção 5)', 'da seção A5)')
              .replace('`code/Dataset.py`', '`projeto_original/code/Dataset.py`').replace('`code/Example.ipynb`', '`projeto_original/code/Example.ipynb`')
              .replace('`Evolution.py` **não foi alterado**', '`projeto_original/code/Evolution.py` **não foi alterado**'))
        c.source = t

decorar(A, 'A'); decorar(B, 'B')

# ---------------------------------------------------------------- 0. Visão geral
md(nb, r"""
# Replicação e melhoria do *Fast Genetic Algorithm for Feature Selection* (census-income)

**Disciplina:** IA Avançada (UNICAP) · **Artigo:** Altarabichi, Nowaczyk, Pashami & Mashhadi, *Expert Systems with Applications* 211 (2023) 118528 · [arXiv:2404.03996](https://arxiv.org/pdf/2404.03996) · **Código oficial:** [Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection](https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection)

Este notebook contém **todo o trabalho em um só lugar**, em duas partes que seguem a tarefa da disciplina:

| Parte | Tarefa | Pergunta que responde |
|---|---|---|
| **A — Replicação** | Reproduzir exatamente o artigo com o código oficial | *Conseguimos obter os mesmos números da Tabela 2 do artigo?* |
| **B — Melhoria** | Propor e implementar uma melhoria, na mesma base | *É possível melhorar o resultado onde o artigo reconhece uma limitação?* |
| **C — Resumo para o relatório** | Respostas diretas ao roteiro da disciplina | *O que escrever no relatório e nos slides?* |

## Como o projeto está organizado

```
Fast-Genetic-Algorithm-For-Feature-Selection-main/
├── projeto_original/            ← o projeto dos AUTORES (só 3 correções de compatibilidade)
│   ├── code/                    Dataset.py, Evolution.py, Example.ipynb
│   ├── data/census-income.data
│   └── results/results.csv
└── nossa_replica/               ← o NOSSO trabalho
    ├── projeto_replica_e_melhoria.ipynb   (este notebook)
    ├── melhoria/DatasetMod.py   a única mudança de lógica da melhoria
    ├── src/                     runner.py (executa as buscas) e analise.py (tabelas/gráficos)
    ├── results/                 runs/*.json (1 por execução) e os CSVs finais
    ├── figures/                 PNGs 150 dpi para os slides
    └── README.md
```

A regra que organiza tudo: **nada em `projeto_original/` muda a lógica dos autores**. A melhoria vive inteira em `nossa_replica/`, numa subclasse.

## Mapa do notebook

| Seção | O que faz | Por que está aqui |
|---|---|---|
| A1–A2 | Explica o problema, o artigo e os 4 algoritmos | Contexto para julgar a replicação |
| A3 | Setup, sementes, correções de compatibilidade | Reprodutibilidade e transparência |
| A4 | Análise exploratória | Descobre o desbalanceamento |
| A5 | Baselines (DT completa e classe majoritária) | Régua de comparação |
| A6 | Roda os 4 algoritmos × `N_RUNS` | Gera os dados da replicação |
| A7 | Tabela 2 vs artigo | Critério de sucesso da replicação |
| A8 | Gráficos | Qualidade, tempo e atributos escolhidos |
| A9 | Divergências artigo × código | Honestidade metodológica |
| A10 | Limitações observadas | **Ponte para a melhoria** |
| A11 | Salva resultados | Insumo do relatório |
| B1 | Motivação da melhoria | Por que mexer |
| B2 | A melhoria em palavras simples + implementação | O que muda e o que não muda |
| B3–B4 | Resultados e testes estatísticos | Quanto melhora, e se é consistente |
| B5 | Gráficos | Trade-off visível |
| B6 | Ablações, MCC e coluna 24 | Isola o efeito de cada ingrediente |
| B7 | Transferência para outros classificadores | A seleção generaliza além da árvore? |
| B8 | Discussão e recomendação | Conclusão |
| C | Resumo para o relatório | Roteiro da disciplina |

## Como executar (e quanto demora)

* Rode o notebook **a partir da pasta `nossa_replica/`**.
* `N_RUNS` (padrão 10) controla as repetições; `N_RUNS=1` é o **modo rápido** para validar o pipeline. Variáveis de ambiente opcionais: `N_RUNS`, `N_RUNS_CHC`, `RODAR_M3`, `RODAR_ABLACOES`, `RODAR_EXTRA` (0/1) e `EXECUTAR` (0 = só lê resultados já salvos, sem rodar buscas).
* Cada execução é gravada em `results/runs/` ao terminar; se o kernel cair, reexecute as células — só o que falta é refeito.
* Tempo aproximado com 8 processos: Parte A ≈ 10–15 min; Parte B ≈ 1–1,5 h sem M3 e ≈ 3 h com M3 (SMOTE). Detalhes no `README.md`.
""")

# ---------------------------------------------------------------- PARTE A
md(nb, r"""
---
# PARTE A — Replicação do artigo

Objetivo: **replicar exatamente** o experimento na base census-income, com o código dos autores e sem alterar sua lógica, e depois examinar criticamente os resultados.
""")
a0 = A[0].source
a0 = a0[a0.index('## A1.'):]
A[0].source = a0
for i, c in enumerate(A):
    if i == len(A) - 1:      # resumo da Parte A → vai para a Parte C
        continue
    if c.cell_type == 'code' and 'RAIZ = Path.cwd().parent' in c.source:
        c.source = c.source.replace(
            "RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(RAIZ / 'src')); sys.path.insert(0, str(RAIZ / 'code'))",
            "RAIZ = Path.cwd()                                   # pasta nossa_replica/ (rode o notebook a partir dela)\n"
            "if RAIZ.name != 'nossa_replica' and (RAIZ / 'nossa_replica').exists():\n    RAIZ = RAIZ / 'nossa_replica'\n"
            "ORIGINAL = RAIZ.parent / 'projeto_original'          # código e dados dos autores\n"
            "sys.path.insert(0, str(RAIZ / 'src')); sys.path.insert(0, str(ORIGINAL / 'code')); sys.path.insert(0, str(RAIZ / 'melhoria'))")
    c.source = c.source.replace("RAIZ / 'data'", "ORIGINAL / 'data'") if c.cell_type == 'code' else c.source
    nb.cells.append(c)
# remove o resumo da Parte A (última célula já foi pulada)

# ---------------------------------------------------------------- PARTE B
md(nb, r"""
---
# PARTE B — Melhoria: otimizar uma fitness que enxergue a classe minoritária

**Mesma base (census-income), mesma divisão 60/20/20, mesmos 4 algoritmos, mesmos hiperparâmetros** da Parte A. A única coisa que muda é *o que a busca otimiza*.
""")
b0 = B[0].source
b0 = b0[b0.index('## B1.'):]
# separa B1 de B2 para inserir a explicação "em palavras simples" entre eles
i2 = b0.index('## B2.')
md(nb, b0[:i2])
md(nb, r"""
## B1.5 A melhoria em palavras simples

> 🎯 **O que esta seção faz:** explica a melhoria sem código, antes de entrarmos na implementação.
>
> 💡 **Por que existe:** para o grupo conseguir explicar a ideia nos slides em poucas frases.

**O problema, numa frase.** Os algoritmos do artigo escolhem atributos para *maximizar a acurácia*. Como 94 % das pessoas da base ganham até 50 mil, um modelo que quase só responde "ganha pouco" já acerta muito — e o algoritmo é **recompensado por isso**. O resultado: subconjuntos de atributos que acertam 94,9 %, mas só encontram **1 em cada 4** pessoas de renda alta (recall ≈ 0,24).

**A melhoria, numa frase.** Trocar a "nota" que a busca tenta maximizar (a *fitness*): em vez de acurácia, usar uma medida que **só dá nota alta se o modelo encontrar a classe rara** — F1 da classe positiva ou acurácia balanceada.

**Analogia.** É como avaliar um vigia de segurança só pela porcentagem de dias "sem incidente": se 94 % dos dias são tranquilos, o vigia que *nunca dispara o alarme* tira nota 94. Trocar a nota para "quantos incidentes reais ele detectou, sem alarmes falsos demais" muda quem é considerado o melhor vigia.

**Como foi implementado (e por que é elegante).** O código dos autores sempre pede a fitness por dois métodos de `Dataset` (`set_validation_accuracy` / `get_validation_accuracy`). Criamos uma **subclasse** `DatasetMod` que sobrescreve apenas esses métodos para usar outra métrica. Efeitos:

1. A fitness dos **quatro** algoritmos muda de uma só vez — sem editar `Evolution.py`.
2. O **meta-modelo do QX** também muda: ele escolhe a amostra de instâncias pela correlação de Spearman entre o ranking do meta-modelo e o ranking da função original; como a "função original" agora é F1 (ou acurácia balanceada), o meta-modelo passa a imitar *essa* métrica. Isso é coerente com a ideia central do artigo.

**As três configurações testadas**

| Config | Muda… | Intuição |
|---|---|---|
| **M1 — F1** | só a métrica de fitness | "Só vale se encontrar positivos *sem* chutar demais" |
| **M2 — acur. balanceada + `class_weight='balanced'`** | métrica **e** o treino da árvore | "Cada classe vale igual; errar um rico pesa ~15× mais" (é a sugestão dos próprios autores: *recall médio*) |
| **M3 — F1 + SMOTE** (opcional) | métrica **e** os dados de treino | "Fabrica exemplos sintéticos de ricos só no treino" |

**O que *não* muda:** base, divisão 60/20/20, normalização, classificador (árvore, `random_state=0`), algoritmos, hiperparâmetros, número de repetições, e o fato de o **teste nunca** participar da busca.

**O que esperar (o trade-off).** A acurácia **vai cair** — é inevitável, porque o original maximiza justamente ela. Em troca, recall, F1 e acurácia balanceada devem subir. A pergunta honesta não é "ficou melhor?", e sim "*a troca vale a pena para o objetivo do problema?*" — discutida na seção B8.
""")
md(nb, b0[i2:])
# setup da Parte B: parâmetros (os imports já foram feitos na Parte A)
for c in B[1:]:
    if c.cell_type == 'code' and 'N_RUNS_CHC' in c.source and 'import os, sys' in c.source:
        c.source = r"""
# ------------------------- PARÂMETROS DA PARTE B -------------------------
# N_RUNS (definido na Parte A) = repetições de PSO/PSO_QX. Para CHC/CHC_QX pode-se usar menos
# repetições se o tempo for curto — nesse caso, deixe isso explícito no relatório.
N_RUNS_CHC = int(os.environ.get('N_RUNS_CHC', N_RUNS))
RODAR_M3 = os.environ.get('RODAR_M3', '1') == '1'          # M3 (SMOTE) é opcional e a mais lenta
RODAR_ABLACOES = os.environ.get('RODAR_ABLACOES', '1') == '1'  # M2b, M3b, M4 (ablações e MCC)
RODAR_EXTRA = os.environ.get('RODAR_EXTRA', '1') == '1'        # experimento extra: sem a coluna 24
EXECUTAR = os.environ.get('EXECUTAR', '1') == '1'              # 0 = só lê o que já está em results/runs (não roda buscas novas)
EXECUTAR_TRANSF = os.environ.get('EXECUTAR_TRANSF', '1' if EXECUTAR else '0') == '1'   # avaliações de transferência (B7)
SEEDS_PSO, SEEDS_CHC = list(range(N_RUNS)), list(range(N_RUNS_CHC))
print(f'N_RUNS (PSO)={N_RUNS} | N_RUNS_CHC={N_RUNS_CHC} | M3={RODAR_M3} | ablações={RODAR_ABLACOES} | extra={RODAR_EXTRA} | executar={EXECUTAR}')
print('\n' + open(RAIZ / 'melhoria' / 'DatasetMod.py').read())
""".strip('\n')
        break
discussao = r"""
**Leitura dos resultados** *(10 sementes por algoritmo e configuração, 360 execuções; medianas no teste. Os testes de Wilcoxon pareados por semente têm n = 10, então o menor p possível é ≈ 0,002 — vários resultados "*" abaixo estão nesse piso.)*

1. **A acurácia cai em todas as configurações que trocam a fitness — e isso é esperado.** O original (≈ 94,9 %) otimiza exatamente a acurácia; M1 e M4 ficam em ≈ 93,0–93,2 %, M3 em ≈ 92,1–92,6 % e M2 em ≈ 81–82 %. M1, M3 e M4 ficam *abaixo* do classificador majoritário (93,7 %), o que só preocupa se o objetivo for acertar o máximo de pessoas; aqui o objetivo é encontrar quem tem renda alta.
2. **M1 (F1) é o melhor equilíbrio.** Nos quatro algoritmos, o recall sobe de ≈ 0,24 para ≈ 0,47–0,48 (Δ ≈ +0,23, p ≈ 0,002 em todos), o F1 de ≈ 0,376 para ≈ 0,466 (Δ ≈ +0,09, p ≈ 0,002) e a acurácia balanceada de ≈ 0,62 para ≈ 0,72 (p ≈ 0,002). A precisão cai de ≈ 0,79–0,84 para ≈ 0,45–0,46. **O MCC, porém, não melhora**: fica em ≈ 0,43 (Δ entre −0,009 e 0, p ≥ 0,08) — o ganho de recall é pago, quase na mesma medida, com perda de precisão. O F1 fica acima do da árvore com os 41 atributos (≈ 0,45), sem alterar o classificador.
3. **M2 (acur. balanceada + `class_weight`) vai ao extremo do recall.** Recall ≈ 0,85–0,87 e acurácia balanceada ≈ 0,84 (a maior de todas), mas a precisão cai para ≈ 0,23–0,25 e o MCC cai (≈ 0,38–0,40, p ≈ 0,002 em todos). O F1 (≈ 0,36–0,38) fica **próximo do original**: menor nos PSO (Δ ≈ −0,013 a −0,017, p < 0,02), sem diferença significativa nos CHC. A ablação **M2b** (mesma acurácia balanceada **sem** `class_weight`) cai no mesmo ponto de M1 (recall ≈ 0,48, F1 ≈ 0,47): o comportamento extremo de M2 vem do **`class_weight`**, não da métrica (F1 de M2b − M2 ≈ +0,08 a +0,11, p ≈ 0,002 nos 4). Faz sentido quando **perder um positivo custa muito mais que um alarme falso**.
4. **M3 (F1 + SMOTE) não compensou.** Recall um pouco maior que o de M1 (≈ 0,50 contra ≈ 0,48; p < 0,03 nos 4), mas F1 e MCC menores nos quatro algoritmos (F1 ≈ 0,446 contra ≈ 0,466; Δ ≈ −0,02; significativo no CHC, p ≈ 0,002, e no PSO pelo teste *t*, mas não no CHC_QX e PSO_QX, p ≈ 0,08–0,38), acurácia menor e custo de tempo ≈ 3,5× (≈ 3.100–3.350 s contra ≈ 860 s nos puros). **M3b** (acurácia + SMOTE) **não ajuda e é instável**: recall e F1 sem ganho significativo no PSO/QX (p ≥ 0,27), *piores* no CHC (F1 −0,06, p ≈ 0,049) e com execuções degeneradas (no CHC, 2 de 10 execuções acertaram a maioria mas encontraram ≈ 0 % dos positivos). O SMOTE sozinho não faz a busca olhar para a classe rara; quem faz isso é a métrica.
5. **MCC (M4) ≈ F1 (M1).** Recall e F1 de M4 não diferem de M1 de forma significativa (recall p ≥ 0,09; F1 p ≥ 0,06). Note que otimizar o MCC **não aumentou o MCC** em relação ao original (≈ 0,43 nos dois; Δ entre −0,009 e +0,008): a base tem um ótimo de MCC já alcançado pela acurácia, o que reforça que MCC e F1 medem trade-offs diferentes — o MCC "não enxerga" o ganho de recall que o F1 enxerga. Para encontrar a classe rara, F1 é a escolha mais simples de explicar.
6. **A coluna 24 (instance weight) não muda nada.** Removê-la (X1, X2) deixa acurácia, recall e F1 iguais aos de `original` e M1 (recall e F1: p ≥ 0,37; Δ ≈ 0). A busca não dependia dela.
7. **O ganho de tempo do QX se mantém.** Tempo total mediano por execução (inclui carga dos dados e reavaliação final; medido com 8 processos em paralelo, então vale como ordem de grandeza): original ≈ 70–75 s (QX) contra ≈ 230–270 s (puros); com F1 ≈ 210–280 s (QX) contra ≈ 860 s (puros). Pelo tempo até a melhor solução (`time_best`), o ganho é ≈ 3,4–3,6× no original e ≈ 2,4–7× com F1. O custo absoluto do F1 é maior que o da acurácia (≈ 3–4× nos QX).
8. **Os subconjuntos mudam.** O original escolhe poucos atributos (5–8); com F1/MCC a busca prefere ≈ 23–35; com M2 ≈ 6–7; com acurácia + SMOTE ≈ 4–5. Ver figuras B5.4/B5.5.
9. **Transferência (B7).** Treinando LR e RF no treino original só com os atributos escolhidos, os subconjuntos de F1/MCC/M2b/M3 dão PR-AUC ≈ 0,53–0,54 (LR) e ≈ 0,61–0,62 (RF) contra ≈ 0,35 e ≈ 0,41 dos subconjuntos do original; ROC-AUC ≈ 0,925–0,942 contra ≈ 0,80–0,87. Ou seja, os subconjuntos maiores não servem só à árvore: carregam informação útil a outros modelos. Os do original (6 atributos) e os de M3b (4) são os piores; M2 (7 atributos) fica no meio.

**Quando usar cada configuração**

| Situação | Recomendação |
|---|---|
| Classes balanceadas, ou acurácia é de fato o objetivo | `original` |
| Compromisso entre achar positivos e não gerar alarmes falsos demais (caso geral) | **M1 (F1)** — M4 (MCC) e M2b equivalem |
| Perder um positivo é muito mais caro que um alarme falso | M2 (acur. balanceada + `class_weight`) |
| Interesse em aumento de dados | M3 — sem vantagem clara e ≈ 3,5× mais lento |

**Recomendação final:** usar **M1 (F1 como fitness)** como substituto direto da acurácia: troca de uma linha, recall dobra, F1 e acurácia balanceada sobem, o ganho do QX permanece. O custo é tempo e subconjuntos maiores (≈ 25–33 atributos em vez de 5–8).

**Limitações da melhoria (para o relatório)**

* A divisão 60/20/20 é **uma só** e sequencial (sem validação cruzada); a variabilidade vem apenas das sementes das buscas.
* A fitness é calculada na validação e a melhor solução é escolhida nela — há alguma tendência de *overfitting* à validação (por isso o teste só é usado no fim).
* Árvore de decisão totalmente crescida com limiar fixo (0,5): outros classificadores ou ajuste do limiar poderiam mudar o trade-off.
* O trade-off precisão × recall é uma escolha de **negócio**, não estatística; mostramos os dois lados.
* Com n = 10, o teste de Wilcoxon tem piso de p ≈ 0,002; diferenças pequenas (como M1 × M3 em F1) não são detectáveis com tão poucas sementes.
* Os tempos foram medidos com 8 processos simultâneos numa máquina de 10 núcleos; servem como ordem de grandeza, não como *benchmark* isolado.
* A baseline (árvore com 41 atributos) dá 92,815 % aqui contra 92,87 % do artigo; a causa não foi identificada.
"""
for c in B[1:]:
    if c.cell_type == 'markdown' and 'DISCUSSAO_PLACEHOLDER' in c.source:
        c.source = discussao.strip('\n')

NOVAS = []
def _call(parte, n):
    faz, pq = CALL[parte][n]
    return f'\n\n> 🎯 **O que esta seção faz:** {faz}\n>\n> 💡 **Por que existe:** {pq}'
def nmd(t): NOVAS.append(nbf.v4.new_markdown_cell(t.strip('\n')))
def ncode(t): NOVAS.append(nbf.v4.new_code_cell(t.strip('\n')))

nmd("## B6. Ablações, MCC e o experimento sem a coluna 24" + _call('B', '6') + r"""

Cada configuração extra responde a uma pergunta específica:

| Pergunta | Comparação | Como ler |
|---|---|---|
| O ganho de M2 vem da **métrica** ou dos **pesos de classe**? | `M2b_bacc` (só métrica) × `M2_bacc_bal` (métrica + pesos) × `original` | Se M2b ≈ original, a métrica sozinha não basta e o ganho vem dos pesos |
| O SMOTE ajuda por si só, ou só junto com o F1? | `M3b_acc_smote` × `original` e `M3_f1_smote` × `M1_f1` | M3b isola o SMOTE sob acurácia |
| F1 ou MCC como fitness? | `M4_mcc` × `M1_f1` | Duas formas de resumir a matriz de confusão |
| A coluna 24 (*instance weight*) estava inflando o resultado? | `X1_acc_sem24` × `original` e `X2_f1_sem24` × `M1_f1` | Se a métrica cai sem ela, a árvore dependia de um atributo que não descreve a pessoa |

As tabelas completas estão em B3 e os testes pareados em B4 (que já incluem esses pares). Abaixo, uma visão compacta das medianas.
""")
ncode(r"""
alvo = {'test_accuracy': 'acurácia', 'test_recall': 'recall', 'test_precision': 'precisão', 'test_f1': 'F1', 'test_balanced_accuracy': 'acur. balanceada'}
for m, nome in alvo.items():
    piv = df.pivot_table(index='config', columns='algo', values=m, aggfunc='median').reindex(CFGS)[ALGS]
    print(f'--- mediana de {nome} (config × algoritmo) ---')
    display((piv * (100 if m == 'test_accuracy' else 1)).round(3 if m != 'test_accuracy' else 2))
""")
ncode(r"""
print('Atributos usados com mais frequência por configuração (índice: % das soluções) — veja se a coluna 24 aparece:')
for c in CFGS:
    top = freq.loc[c].sort_values(ascending=False).head(6)
    print(f'  {c:14s}', {int(k): int(round(v)) for k, v in top.items()})
""")

nmd("## B7. Transferência: os subconjuntos escolhidos servem para outros classificadores?" + _call('B', '7') + r"""

A busca usa **sempre** uma árvore de decisão. Mas se um subconjunto de atributos é realmente bom, ele deveria ajudar outros modelos também. Então pegamos o subconjunto final de cada execução e treinamos, **no treino original (sem SMOTE e sem pesos da árvore)**:

* **Regressão Logística** (`class_weight='balanced'`) — modelo linear;
* **Random Forest** (30 árvores, `class_weight='balanced_subsample'`) — conjunto de árvores.

Medimos no **teste**: **ROC-AUC** e **PR-AUC** (não dependem de limiar; a PR-AUC é a mais informativa com classes desbalanceadas), além de recall e F1 no limiar 0,5. Ficam como referência os mesmos modelos com **todos os 41 atributos**. Nenhuma busca é refeita; só se treina um modelo por subconjunto.
""")
ncode(r"""
transf = runner.transferencia_todas(df, n_jobs=N_JOBS, executar=EXECUTAR_TRANSF)
transf = transf[transf.config.isin(CFGS)]
# referência: todos os 41 atributos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score
dsb = runner._dataset_base(); ref = {}
for nome, m in {'LR': LogisticRegression(max_iter=300, class_weight='balanced'),
                'RF': RandomForestClassifier(n_estimators=30, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=N_JOBS, random_state=0)}.items():
    m.fit(dsb.X_train, dsb.y_train); p = m.predict_proba(dsb.X_test)[:, 1]
    ref[nome] = dict(roc_auc=roc_auc_score(dsb.y_test, p), pr_auc=average_precision_score(dsb.y_test, p),
                     recall=recall_score(dsb.y_test, p >= .5), f1=f1_score(dsb.y_test, p >= .5))
print('Referência (41 atributos):', {k: {m: round(v, 3) for m, v in d.items()} for k, d in ref.items()})
tab = transf.groupby('config')[[f'{m}_{k}' for m in ('LR', 'RF') for k in ('pr_auc', 'roc_auc', 'recall', 'f1')]].median().reindex(CFGS).round(3)
display(tab)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=False)
for ax, m in zip(axs, ['LR', 'RF']):
    sns.boxplot(data=transf, x='config', y=f'{m}_pr_auc', order=CFGS, hue='config', hue_order=CFGS, palette=[CORES[c] for c in CFGS], legend=False, ax=ax, width=.6, fliersize=0)
    sns.stripplot(data=transf, x='config', y=f'{m}_pr_auc', order=CFGS, color='#222', size=2.5, ax=ax)
    ax.axhline(ref[m]['pr_auc'], ls='--', c='#888'); ax.text(len(CFGS) - .5, ref[m]['pr_auc'], ' 41 atributos', va='bottom', ha='right', fontsize=8, color='#555')
    ax.set_title({'LR': 'Regressão Logística', 'RF': 'Random Forest'}[m]); ax.set_xlabel(''); ax.set_ylabel('PR-AUC no teste')
    ax.set_xticks(range(len(CFGS))); ax.set_xticklabels([c.split('_')[0] for c in CFGS])
plt.tight_layout(); salvar_fig(fig, 'mel_06_transferencia_pr_auc'); plt.show()
""")

for c in B[1:]:
    if c.cell_type == 'markdown' and c.source.startswith('## B8.'):
        for n in NOVAS: nb.cells.append(n)
        NOVAS.clear()
    if c.cell_type == 'code' and "melhoria_runs.csv" in c.source:
        c.source = c.source.replace("print('salvos:", "transf.to_csv(RES_DIR / 'melhoria_transferencia.csv', index=False)\nprint('salvos:")
    nb.cells.append(c)

# ---------------------------------------------------------------- C. Resumo
md(nb, r"""
---
# PARTE C — Resumo para o relatório

> 🎯 **O que esta seção faz:** responde, item a item, ao roteiro da disciplina.
> 💡 **Por que existe:** o grupo pode copiar daqui para o relatório e os slides.

* **Descrição da base:** census-income (UCI Census-Income KDD) — 199.523 instâncias × 41 atributos numéricos já codificados + rótulo binário (renda > 50K), sem valores faltantes. **Desbalanceada: 93,79 % classe 0 e 6,21 % classe 1.** A coluna 24 é o *instance weight* (não deveria ser atributo de entrada; o código a usa como atributo).
* **Tamanho da população:** 50 indivíduos (CHC) / 50 partículas (PSO), nos quatro algoritmos. Meta-modelo QX: `q = 10`, controle de evolução a cada `f = 10` gerações, `f_no_change = 2`.
* **% treino/validação/teste:** 60 % / 20 % / 20 %, **sequencial, sem embaralhar** (`shuffle=False`), normalização *z-score* ajustada só no treino. Fitness na validação; teste só na avaliação final. (SMOTE, em M3, só no treino.)
* **Validação cruzada:** **não é usada.** O artigo usa *holdout* 60/20/20; a robustez estatística vem de **repetições independentes** (sementes 0 a N−1; artigo: 10) dos algoritmos estocásticos, com mediana ± desvio padrão. Na Parte B, os testes de Wilcoxon e *t* são pareados por semente.
* **Algoritmos:** CHC_QX, PSO_QX, CHC e PSO — seleção de atributos *wrapper* com Árvore de Decisão (`random_state=0`).
* **Critério dos hiperparâmetros:** valores do artigo/notebook oficial, **sem *Grid Search***. Divergências conhecidas entre artigo e código (q = 20 vs 10; tempo de CPU do CHC; embaralhamento) foram documentadas e mantivemos o que o código faz. As métricas de fitness da melhoria foram escolhidas por justificativa (B2), nunca pelo desempenho no teste.
* **Transfer learning / *dropout*:** **não se aplicam** — não há rede neural nem pré-treinamento; base tabular e modelo de árvore de decisão.
* **Aumento de dados:** não faz parte do artigo; aparece apenas em M3 (SMOTE no treino), como alternativa de tratamento do desbalanceamento.
* **Desbalanceamento:** o original **não trata** (a acurácia esconde um recall ≈ 0,24); a melhoria **trata**, trocando a fitness por F1 (M1) ou acurácia balanceada com `class_weight` (M2), ou aumentando os dados (M3).
* **Métricas:** acurácia (a do artigo), acurácia balanceada, precisão, recall e F1 da classe 1, MCC, ROC-AUC, matriz de confusão, nº de atributos e tempo até a melhor solução.
* **Resultado da replicação:** ver Tabela da seção A7 (nossa mediana × artigo, Δ em pp).
* **Resultado da melhoria:** ver tabela B3 e testes B4; ablações em B6, transferência em B7; recomendação em B8 (M1 como padrão; M2 se recall for prioritário). Com 10 sementes: recall ≈ 0,24 → ≈ 0,48 e F1 ≈ 0,376 → ≈ 0,466 com M1, ao custo de ≈ 1,8 pp de acurácia; M2 chega a recall ≈ 0,86 com precisão ≈ 0,24; M3 (SMOTE) é ≈ 3,5× mais lento sem ganho de F1; a coluna 24 não altera nada.
""")

salvar(nb, '../projeto_replica_e_melhoria.ipynb')
