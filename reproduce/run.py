"""Run software checks, verify funding estimates, and regenerate manuscript figures."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

run("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider")
run("reproduction/verify_funding.py", "--report", "reproduction/data/funding_verification.json")
run("reproduction/make_journal_figures.py")
print("Software, funding-statistic and manuscript-figure reproduction completed.")
