import nbformat as nbf
def novo(): return nbf.v4.new_notebook()
def md(nb, s): nb.cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(nb, s): nb.cells.append(nbf.v4.new_code_cell(s.strip('\n')))
def salvar(nb, p):
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbf.write(nb, p)
