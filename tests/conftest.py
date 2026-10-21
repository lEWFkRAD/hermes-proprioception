"""Make the plugin importable as ``plugins.proprioception``.

The plugin's modules import each other as ``plugins.proprioception.*`` — the
same package path the real Hermes loader (hermes_cli/plugins.py:
_load_directory_module) gives a directory plugin. Reproduce that here with
spec_from_file_location + submodule_search_locations so the tests exercise
this checkout exactly as Hermes would load it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]

if "plugins" not in sys.modules:
    try:
        import plugins  # noqa: F401  — real package when running inside hermes-agent
    except ImportError:
        _pkg = types.ModuleType("plugins")  # standalone repo: synthesize the parent
        _pkg.__path__ = []
        sys.modules["plugins"] = _pkg

if "plugins.proprioception" not in sys.modules:

    _spec = importlib.util.spec_from_file_location(
        "plugins.proprioception",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    sys.modules["plugins.proprioception"] = _mod
    _spec.loader.exec_module(_mod)
