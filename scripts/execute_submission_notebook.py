"""Run all presentation cells and export an offline HTML reading copy."""
from pathlib import Path
import json
import os
import sys

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/consumer_complaint_classification.ipynb"


def main():
    notebook = nbformat.read(PATH, as_version=4)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell {i}", "exec")
    # Start in a folder without the project's source/data to verify the embedded
    # report path is portable. Optional live inference remains explicitly off.
    work = ROOT / ".qa/notebook-portability"
    work.mkdir(parents=True, exist_ok=True)
    # Windows sandbox ACLs prevent Jupyter from creating its secure connection
    # file. Execute the same code cells in a fresh headless IPython process,
    # without sockets or changes to Jupyter's connection-file protections.
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.qa/matplotlib')
    os.environ['IPYTHONDIR'] = str(ROOT / '.qa/ipython')
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from matplotlib_inline.backend_inline import configure_inline_support, InlineBackend
    shell = InteractiveShell.instance(config=Config({'HistoryManager': {'hist_file': ':memory:'}}))
    configure_inline_support(shell, 'inline')
    InlineBackend.instance().figure_formats = {'png'}
    os.chdir(work)
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output() as captured:
            executed = shell.run_cell(cell.source, store_history=True)
        if executed.error_before_exec or executed.error_in_exec:
            raise RuntimeError(f'Cell {index} failed: {executed.error_before_exec or executed.error_in_exec}')
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        cell.outputs = outputs
        cell.execution_count = count
        print(f'Executed cell {index}', flush=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    # Keep compressed evidence available inside the notebook, while excluding
    # its non-teaching bootstrap cell from the HTML reading copy.
    exporter.register_preprocessor('nbconvert.preprocessors.TagRemovePreprocessor', enabled=True)
    exporter.config.TagRemovePreprocessor.remove_input_tags = {'hide-input'}
    body, _ = exporter.from_notebook_node(notebook)
    html_path = PATH.with_suffix(".html")
    html_path.write_text(body, encoding="utf-8")
    image_outputs = sum("image/png" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", []))
    code_cells = sum(c.cell_type == "code" for c in notebook.cells)
    assert image_outputs >= 7
    result = {"status": "passed", "cells": len(notebook.cells), "code_cells": code_cells,
              "execution_engine": "Headless IPython in a fresh Python process (Jupyter socket launch blocked by Windows ACLs)",
              "executed_code_cells": sum(c.get("execution_count") is not None for c in notebook.cells if c.cell_type == "code"),
              "inline_chart_outputs": image_outputs, "executed_from_independent_folder": True,
              "model_training_performed": False, "data_downloads_performed": False,
              "notebook_bytes": PATH.stat().st_size, "html_bytes": html_path.stat().st_size}
    (ROOT / "reports/notebook_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
