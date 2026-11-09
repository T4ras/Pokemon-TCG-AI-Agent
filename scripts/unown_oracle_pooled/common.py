from pathlib import Path
from experiment_supervisor.io import read,atomic as write,sha,utc as iso
ROOT=Path(__file__).resolve().parents[2]
RUN='astra-unown-belief-correction-pooled-evaluation-001'
OUT=ROOT/'experiments'/RUN
