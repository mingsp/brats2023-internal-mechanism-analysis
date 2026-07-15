import importlib.metadata

import pptt


def test_package_import_and_version():
    assert pptt.__version__ == "0.1.0"
    assert importlib.metadata.version("pptt-process-xai") == pptt.__version__
