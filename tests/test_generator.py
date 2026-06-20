"""
Tests del core de generación (T-007).
No requieren [agents] ni conexión a LLM.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from gen_peru_poblacion.config import Config
from gen_peru_poblacion.generator import PopulationGenerator
from gen_peru_poblacion.verify import run_ks_check

FUENTE_DIR = Path(__file__).parent.parent / "data" / "fuentes"

REQUIRED_FIELDS = {
    "perfil_id",
    "synthetic",
    "data_sources",
    "region",
    "rubro",
    "tamaño",
    "formalizado",
    "canal_venta",
    "adopcion_digital",
    "credito",
    "edad_dueño",
    "nivel_educativo",
    "ingreso_mensual_soles",
    "lengua_materna",
}


@pytest.fixture(scope="module")
def generator():
    cfg = Config(segmento="mype", n=10, fuente_dir=str(FUENTE_DIR))
    return PopulationGenerator(cfg)


@pytest.fixture(scope="module")
def perfiles_10(generator):
    return generator.generate()


def test_generate_profiles_basic(perfiles_10):
    """Genera 10 perfiles y verifica que el resultado es una lista de dicts."""
    assert isinstance(perfiles_10, list)
    assert len(perfiles_10) == 10
    assert all(isinstance(p, dict) for p in perfiles_10)


def test_profiles_have_required_fields(perfiles_10):
    """Todos los perfiles contienen exactamente los campos requeridos."""
    for perfil in perfiles_10:
        missing = REQUIRED_FIELDS - set(perfil.keys())
        assert not missing, f"Faltan campos: {missing} en perfil {perfil.get('perfil_id')}"


def test_synthetic_flag(perfiles_10):
    """Todos los perfiles tienen synthetic=True."""
    for perfil in perfiles_10:
        assert perfil["synthetic"] is True, f"synthetic debe ser True, got {perfil['synthetic']}"


def test_unique_perfil_ids(perfiles_10):
    """No hay UUIDs duplicados entre perfiles."""
    ids = [p["perfil_id"] for p in perfiles_10]
    assert len(set(ids)) == len(ids), "Hay perfil_id duplicados"
    for pid in ids:
        uuid.UUID(pid)  # Lanza ValueError si no es UUID válido


def test_export_csv(generator, perfiles_10, tmp_path):
    """El CSV exportado es válido y parseable con pandas."""
    dest = tmp_path / "mype.csv"
    path = generator.export(perfiles_10, formato="csv", path=dest)

    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 10
    assert "perfil_id" in df.columns
    assert "synthetic" in df.columns
    assert "region" in df.columns


def test_export_jsonl(generator, perfiles_10, tmp_path):
    """El JSONL exportado tiene exactamente N líneas, cada una JSON válido."""
    dest = tmp_path / "mype.jsonl"
    path = generator.export(perfiles_10, formato="jsonl", path=dest)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    for line in lines:
        obj = json.loads(line)
        assert "perfil_id" in obj
        assert obj["synthetic"] is True


def test_ks_basic(perfiles_10, generator, tmp_path):
    """El KS score global se puede calcular sin errores y devuelve KSReport."""
    from gen_peru_poblacion.verify import KSReport

    dest = tmp_path / "mype.jsonl"
    generator.export(perfiles_10, formato="jsonl", path=dest)

    report = run_ks_check(
        data_dir=tmp_path,
        fuente_dir=str(FUENTE_DIR),
        segmento="mype",
    )
    assert isinstance(report, KSReport)
    assert 0.0 <= report.global_score <= 1.0
    assert len(report.scores) > 0
    assert report.n_generated == 10
