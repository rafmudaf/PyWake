import os
import sys
import warnings
from pathlib import Path

import pytest
import xarray

import py_wake
from py_wake.flow_map import Grid
import py_wake.tests.notebook as notebook_module
import py_wake.tests.test_files as test_files_module
from py_wake.tests.notebook import Notebook


def get_notebooks():
    def get(path):
        return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]
    path = os.path.dirname(py_wake.__file__) + "/../docs/notebooks/"
    return get(path) + get(path + "exercises/")


notebooks = get_notebooks()
slow_notebooks = {'Shadow.ipynb', 'Site.ipynb', 'ExternalWindFarms.ipynb'}


@pytest.mark.parametrize(
    "notebook",
    [pytest.param(nb, marks=pytest.mark.slow) if os.path.basename(nb.filename) in slow_notebooks else nb
     for nb in notebooks],
    ids=[os.path.basename(nb.filename) for nb in notebooks])
def test_notebooks(notebook, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    if (str(Path(notebook.filename).relative_to(os.path.dirname(py_wake.__file__) + "/../docs/notebooks/")) in
            ['Optimization.ipynb']):
        return

    monkeypatch.chdir(tmp_path)
    notebook_module.tfp = str(tmp_path) + '/'
    monkeypatch.setattr(test_files_module, '__path__', [str(tmp_path)])
    if 'py_wake.tests.test_files.tmp' in sys.modules:
        sys.modules['py_wake.tests.test_files.tmp'].__path__ = [str(tmp_path / 'tmp')]

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    try:
        # print(notebook.filename)
        default_resolution = Grid.default_resolution
        Grid.default_resolution = 100
        plt.rcParams.update({'figure.max_open_warning': 0})
        from py_wake.utils import profiling

        def dummy_profileit(f, *args, **kwargs):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs), 0, 0
            return wrapper
        profiling.profileit = dummy_profileit

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The .* model is not representative of the setup used in the literature')
            notebook.check_code()
        notebook.check_links()
        # notebook.check_pip_header()
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
    finally:
        Grid.default_resolution = default_resolution
        plt.close('all')
        plt.rcParams.update({'figure.max_open_warning': 20})
        xarray.set_options(display_expand_data=True)


if __name__ == '__main__':
    from tempfile import TemporaryDirectory

    # print("\n".join([f.filename for f in get_notebooks()]))
    path = os.path.dirname(py_wake.__file__) + "/../docs/notebooks/"
    f = 'RotorAverageModels.ipynb'
    with TemporaryDirectory() as tmp_dir, pytest.MonkeyPatch.context() as monkeypatch:
        test_notebooks(Notebook(path + f), Path(tmp_dir), monkeypatch)
