"""Executa um notebook com o kernel do venv e restaura o kernelspec genérico (python3)."""
import sys, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
p = sys.argv[1]
nb = nbformat.read(p, as_version=4)
ExecutePreprocessor(timeout=-1, kernel_name='fastga').preprocess(nb, {'metadata': {'path': '.'}})
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.write(nb, p)
