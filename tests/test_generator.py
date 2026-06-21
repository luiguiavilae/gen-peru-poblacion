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


# ══════════════════════════════════════════════════════════════════════════════
# CONSUMIDORES
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS_CONSUMIDORES = {
    "perfil_id", "synthetic", "data_sources", "segmento",
    "region", "nivel_socioeconomico", "nivel_educativo",
    "dispositivo_principal", "marketplace_preferido", "metodo_pago_preferido",
    "lengua_materna", "ocupacion", "sexo", "frecuencia_uso_internet",
    "edad", "ingreso_mensual_soles",
}


@pytest.fixture(scope="module")
def generator_consumidores():
    cfg = Config(segmento="consumidores", n=10, fuente_dir=str(FUENTE_DIR))
    return PopulationGenerator(cfg)


@pytest.fixture(scope="module")
def perfiles_consumidores(generator_consumidores):
    return generator_consumidores.generate()


def test_generate_consumidores_basic(perfiles_consumidores):
    assert isinstance(perfiles_consumidores, list)
    assert len(perfiles_consumidores) == 10
    assert all(isinstance(p, dict) for p in perfiles_consumidores)


def test_consumidores_required_fields(perfiles_consumidores):
    for perfil in perfiles_consumidores:
        missing = REQUIRED_FIELDS_CONSUMIDORES - set(perfil.keys())
        assert not missing, f"Faltan campos: {missing}"


def test_consumidores_synthetic_flag(perfiles_consumidores):
    for p in perfiles_consumidores:
        assert p["synthetic"] is True


def test_consumidores_unique_ids(perfiles_consumidores):
    ids = [p["perfil_id"] for p in perfiles_consumidores]
    assert len(set(ids)) == len(ids)
    for pid in ids:
        uuid.UUID(pid)


def test_consumidores_segmento_field(perfiles_consumidores):
    for p in perfiles_consumidores:
        assert p["segmento"] == "consumidores"


def test_consumidores_export_csv(generator_consumidores, perfiles_consumidores, tmp_path):
    dest = tmp_path / "consumidores.csv"
    path = generator_consumidores.export(perfiles_consumidores, formato="csv", path=dest)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 10
    assert "perfil_id" in df.columns
    assert "nivel_socioeconomico" in df.columns


def test_consumidores_ks_basic(perfiles_consumidores, generator_consumidores, tmp_path):
    from gen_peru_poblacion.verify import KSReport

    dest = tmp_path / "consumidores.jsonl"
    generator_consumidores.export(perfiles_consumidores, formato="jsonl", path=dest)

    report = run_ks_check(
        data_dir=tmp_path,
        fuente_dir=str(FUENTE_DIR),
        segmento="consumidores",
        variables=["region", "nivel_socioeconomico", "nivel_educativo"],
    )
    assert isinstance(report, KSReport)
    assert 0.0 <= report.global_score <= 1.0
    assert report.n_generated == 10


# ══════════════════════════════════════════════════════════════════════════════
# FINANCIERO
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS_FINANCIERO = {
    "perfil_id", "synthetic", "data_sources", "segmento",
    "region", "tipo_entidad_principal", "nivel_bancarizacion",
    "canal_preferido", "nivel_educativo", "lengua_materna",
    "ocupacion", "sexo", "edad", "ingreso_mensual_soles",
}


@pytest.fixture(scope="module")
def generator_financiero():
    cfg = Config(segmento="financiero", n=10, fuente_dir=str(FUENTE_DIR))
    return PopulationGenerator(cfg)


@pytest.fixture(scope="module")
def perfiles_financiero(generator_financiero):
    return generator_financiero.generate()


def test_generate_financiero_basic(perfiles_financiero):
    assert isinstance(perfiles_financiero, list)
    assert len(perfiles_financiero) == 10
    assert all(isinstance(p, dict) for p in perfiles_financiero)


def test_financiero_required_fields(perfiles_financiero):
    for perfil in perfiles_financiero:
        missing = REQUIRED_FIELDS_FINANCIERO - set(perfil.keys())
        assert not missing, f"Faltan campos: {missing}"


def test_financiero_synthetic_flag(perfiles_financiero):
    for p in perfiles_financiero:
        assert p["synthetic"] is True


def test_financiero_unique_ids(perfiles_financiero):
    ids = [p["perfil_id"] for p in perfiles_financiero]
    assert len(set(ids)) == len(ids)
    for pid in ids:
        uuid.UUID(pid)


def test_financiero_segmento_field(perfiles_financiero):
    for p in perfiles_financiero:
        assert p["segmento"] == "financiero"


def test_financiero_export_jsonl(generator_financiero, perfiles_financiero, tmp_path):
    dest = tmp_path / "financiero.jsonl"
    path = generator_financiero.export(perfiles_financiero, formato="jsonl", path=dest)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    for line in lines:
        obj = json.loads(line)
        assert obj["synthetic"] is True
        assert "nivel_bancarizacion" in obj


def test_financiero_ks_basic(perfiles_financiero, generator_financiero, tmp_path):
    from gen_peru_poblacion.verify import KSReport

    dest = tmp_path / "financiero.jsonl"
    generator_financiero.export(perfiles_financiero, formato="jsonl", path=dest)

    report = run_ks_check(
        data_dir=tmp_path,
        fuente_dir=str(FUENTE_DIR),
        segmento="financiero",
        variables=["region", "nivel_bancarizacion", "nivel_educativo"],
    )
    assert isinstance(report, KSReport)
    assert 0.0 <= report.global_score <= 1.0
    assert report.n_generated == 10


# ══════════════════════════════════════════════════════════════════════════════
# SALUD
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS_SALUD = {
    "perfil_id", "synthetic", "data_sources", "segmento",
    "region", "tipo_seguro", "zona", "establecimiento_preferido",
    "motivo_consulta_principal", "nivel_educativo", "lengua_materna",
    "sexo", "edad", "ingreso_mensual_soles",
}


@pytest.fixture(scope="module")
def generator_salud():
    cfg = Config(segmento="salud", n=10, fuente_dir=str(FUENTE_DIR))
    return PopulationGenerator(cfg)


@pytest.fixture(scope="module")
def perfiles_salud(generator_salud):
    return generator_salud.generate()


def test_generate_salud_basic(perfiles_salud):
    assert isinstance(perfiles_salud, list)
    assert len(perfiles_salud) == 10
    assert all(isinstance(p, dict) for p in perfiles_salud)


def test_salud_required_fields(perfiles_salud):
    for perfil in perfiles_salud:
        missing = REQUIRED_FIELDS_SALUD - set(perfil.keys())
        assert not missing, f"Faltan campos: {missing}"


def test_salud_synthetic_flag(perfiles_salud):
    for p in perfiles_salud:
        assert p["synthetic"] is True


def test_salud_unique_ids(perfiles_salud):
    ids = [p["perfil_id"] for p in perfiles_salud]
    assert len(set(ids)) == len(ids)
    for pid in ids:
        uuid.UUID(pid)


def test_salud_segmento_field(perfiles_salud):
    for p in perfiles_salud:
        assert p["segmento"] == "salud"


def test_salud_export_csv(generator_salud, perfiles_salud, tmp_path):
    dest = tmp_path / "salud.csv"
    path = generator_salud.export(perfiles_salud, formato="csv", path=dest)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 10
    assert "tipo_seguro" in df.columns
    assert "zona" in df.columns


def test_salud_ks_basic(perfiles_salud, generator_salud, tmp_path):
    from gen_peru_poblacion.verify import KSReport

    dest = tmp_path / "salud.jsonl"
    generator_salud.export(perfiles_salud, formato="jsonl", path=dest)

    report = run_ks_check(
        data_dir=tmp_path,
        fuente_dir=str(FUENTE_DIR),
        segmento="salud",
        variables=["region", "tipo_seguro", "nivel_educativo"],
    )
    assert isinstance(report, KSReport)
    assert 0.0 <= report.global_score <= 1.0
    assert report.n_generated == 10
