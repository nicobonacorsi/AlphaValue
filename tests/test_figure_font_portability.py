"""Exercise the public figure CLI with and without proprietary system fonts."""
import builtins
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "reproduction/make_journal_figures.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("journal_figure_portability", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_inputs(destination):
    for relative in ("reproduction/reference_results.json", "research/near_critical/two_look_results.csv"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_check_only_needs_no_matplotlib_or_font(monkeypatch, tmp_path):
    generator = load_generator()
    copy_inputs(tmp_path)
    original_import = builtins.__import__

    def block_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError("--check-only must not import Matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_matplotlib)
    monkeypatch.setattr(sys, "argv", [str(GENERATOR), "--root", str(tmp_path), "--check-only", "--font", "times"])
    generator.main()
    assert (tmp_path / "reproduction/journal_figure_values.json").is_file()
    assert not (tmp_path / "reproduction/journal_figure_font_audit.json").exists()
    assert not (tmp_path / "manuscript_source").exists()


def test_missing_times_renders_all_figures_with_explicit_stix_audit(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib", reason="Rendering requires the optional plot dependencies")
    from matplotlib import font_manager

    generator = load_generator()
    copy_inputs(tmp_path)
    original_findfont = font_manager.findfont

    def missing_times(properties, *args, **kwargs):
        families = properties.get_family() if hasattr(properties, "get_family") else [str(properties)]
        if any("Times New Roman" in family for family in families):
            raise ValueError("Times intentionally unavailable for this portability test")
        return original_findfont(properties, *args, **kwargs)

    # Simulate an absent system family even on a Windows developer machine.
    # All STIX lookup and rendering still use the actual Matplotlib font files.
    monkeypatch.setattr(font_manager, "findfont", missing_times)
    with pytest.raises(RuntimeError, match="--font times requires Times New Roman"):
        generator.select_figure_font("times")
    selected = generator.select_figure_font("stix")
    assert selected["selected_mode"] == "stix"
    assert selected["fallback_used"] is False

    monkeypatch.setattr(sys, "argv", [str(GENERATOR), "--root", str(tmp_path)])
    with pytest.warns(RuntimeWarning, match="Times New Roman is unavailable; rendering with STIXGeneral"):
        generator.main()
    audit = json.loads((tmp_path / "reproduction/journal_figure_font_audit.json").read_text())
    assert audit["requested_mode"] == "auto"
    assert audit["selected_mode"] == "stix"
    assert audit["family"] == "STIXGeneral"
    assert audit["mathtext_fontset"] == "stix"
    assert audit["fallback_used"] is True
    assert all(face["name"] == "STIXGeneral" for face in audit["faces"].values())

    names = [("nearcritical", "near_critical"), ("survival_frontier", "survival_frontier"),
             ("equilibrium_competition", "equilibrium_competition"), ("lifetime_benchmark", "lifetime_benchmark")]
    for pdf_stem, svg_stem in names:
        pdf = (tmp_path / f"manuscript_source/fig_{pdf_stem}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-")
        font_names = re.findall(rb"/BaseFont\s+/([^\s/>]+)", pdf)
        assert font_names and all(b"STIX" in name for name in font_names)
        assert b"/FontFile2" in pdf
        assert b"Text uses STIXGeneral and mathematics uses STIX." in pdf
        svg = (tmp_path / f"docs/assets/{svg_stem}.svg").read_text()
        assert "Text uses STIXGeneral and mathematics uses STIX." in svg
        assert "STIX" in svg and "TimesNewRoman" not in svg
        assert (tmp_path / f"reproduction/figure_previews/{svg_stem}.png").is_file()
