import sys
from unittest.mock import patch

import pytest


def test_deprecated_dump_emits_warning():
    from textrl.cli import _deprecated_dump

    with patch("textrl.cli.merge_main", return_value=0) as m:
        with pytest.warns(DeprecationWarning):
            rc = _deprecated_dump(["--rl", "adapter_dir", "--dump", "out"])
        assert rc == 0
        m.assert_called_once()
        args = m.call_args[0][0]
        assert "--adapter" in args and "adapter_dir" in args
        assert "--output" in args and "out" in args


def test_train_main_requires_config():
    from textrl.cli import train_main

    with pytest.raises(SystemExit):
        train_main([])
