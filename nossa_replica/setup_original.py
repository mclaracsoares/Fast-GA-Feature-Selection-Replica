"""Baixa o código e os dados oficiais dos autores para ../projeto_original e aplica as 3 correções
de compatibilidade documentadas no notebook (nenhuma altera a lógica):

  1. Dataset.py : np.NaN                      -> np.nan                 (removido no numpy 2)
  2. Dataset.py : delim_whitespace=True       -> sep=r"\\s+"            (descontinuado no pandas)
  3. Example.ipynb: remove a linha np.warnings.filterwarnings(...)      (np.warnings não existe mais)

Evolution.py NÃO é alterado. Uso (a partir de nossa_replica/):  python setup_original.py
Repositório oficial: https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

URL = 'https://github.com/Ghaith81/Fast-Genetic-Algorithm-For-Feature-Selection.git'
COMMIT = '6ac5a7ec77f8a7c096ab4d019254fcc897988fd6'   # versão usada na replicação
DESTINO = Path(__file__).resolve().parent.parent / 'projeto_original'

if DESTINO.exists():
    sys.exit(f'{DESTINO} já existe; nada a fazer (apague a pasta para baixar de novo).')

with tempfile.TemporaryDirectory() as tmp:
    clone = Path(tmp) / 'oficial'
    subprocess.run(['git', 'clone', '-q', URL, str(clone)], check=True)
    subprocess.run(['git', '-C', str(clone), 'checkout', '-q', COMMIT], check=True)
    shutil.rmtree(clone / '.git')
    shutil.copytree(clone, DESTINO)

# --- correções de compatibilidade (cada uma é conferida; falha em voz alta se o arquivo for outro)
ds = DESTINO / 'code' / 'Dataset.py'
txt = ds.read_text()
for antes, depois in [('np.NaN', 'np.nan'), ('delim_whitespace=True', 'sep=r"\\s+"')]:
    assert antes in txt, f'{antes!r} não encontrado em Dataset.py'
    txt = txt.replace(antes, depois)
ds.write_text(txt)

nbp = DESTINO / 'code' / 'Example.ipynb'
nb = json.loads(nbp.read_text())
removidas = 0
for cel in nb['cells']:
    if cel['cell_type'] == 'code':
        novo = [l for l in cel['source'] if 'np.warnings.filterwarnings' not in l]
        removidas += len(cel['source']) - len(novo)
        cel['source'] = novo
assert removidas == 1, f'esperava remover 1 linha de np.warnings, removi {removidas}'
nbp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print(f'Pronto: {DESTINO} (código oficial no commit {COMMIT[:7]} + 3 correções de compatibilidade).')
