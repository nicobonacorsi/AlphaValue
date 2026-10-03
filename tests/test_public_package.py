from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_public_release_layout():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "pyproject.toml",
        ROOT / "CITATION.cff",
        ROOT / "paper" / "Certified_Alpha_Capacity.pdf",
        ROOT / "docs" / "index.html",
        ROOT / "reproduction" / "README.md",
    ]
    for path in required:
        assert path.exists(), path
    assert not any((ROOT / "paper").glob("*.tex"))

def test_public_release_has_no_private_or_cache_trees():
    forbidden = {"private", "internal", "scratch", ".pytest_cache"}
    present = {p.name.lower() for p in ROOT.iterdir() if p.is_dir()}
    assert not (forbidden & present)


def test_reproduction_reference_files_exist():
    required = [
        ROOT / "reproduction" / "reference_results.json",
        ROOT / "reproduction" / "make_empirical_figures.py",
        ROOT / "reproduction" / "nearcritical_figure_values.json",
    ]
    for path in required:
        assert path.exists(), path
