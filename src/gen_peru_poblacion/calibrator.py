"""
Calibrator: carga distribuciones JSON fuente, construye un dataset de entrenamiento
con correlaciones inducidas (Iman-Conover) y fitea GaussianCopulaSynthesizer de SDV.

Flujo:
  1. _build_training_data(): muestrea marginals → induce correlaciones → mapea a strings
  2. fit(): llama al paso 1, configura metadata SDV, fitea y cachea el modelo
  3. sample(n): genera n perfiles sintéticos desde el modelo fiteado
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import linalg as la

SEGMENTO_FUENTE: dict[str, str] = {
    "mype": "mype_distribucion_2023.json",
    "consumidores": "consumidores_distribucion_2023.json",
    "financiero": "financiero_distribucion_2023.json",
    "salud": "salud_distribucion_2023.json",
}

# ─── Ordinales MYPE ────────────────────────────────────────────────────────────

ADOPCION_ORD = {"nula": 0, "baja": 1, "media": 2, "alta": 3}
ADOPCION_ORD_INV = {v: k for k, v in ADOPCION_ORD.items()}

TAMAÑO_ORD = {"unipersonal": 0, "familiar_2_a_4": 1, "pequena_5_a_10": 2, "mas_de_10": 3}
TAMAÑO_ORD_INV = {v: k for k, v in TAMAÑO_ORD.items()}

EDUCACION_ORD = {
    "sin_nivel": 0,
    "primaria_incompleta": 1,
    "primaria_completa": 2,
    "secundaria_incompleta": 3,
    "secundaria_completa": 4,
    "tecnica_incompleta": 5,
    "tecnica_completa": 6,
    "universitaria_incompleta": 7,
    "universitaria_completa": 8,
}
EDUCACION_ORD_INV = {v: k for k, v in EDUCACION_ORD.items()}

# Orden de columnas en la matriz IC de MYPE (NO cambiar sin actualizar _build_corr_matrix_mype)
_IC_VARS = [
    "edad_num",           # 0
    "nivel_edu_num",      # 1
    "adopcion_num",       # 2
    "formalizado_num",    # 3
    "credito_formal_num", # 4
    "tamaño_num",         # 5
    "ingreso_num",        # 6
    "region_lima_num",    # 7
    "canal_whatsapp_num", # 8
]

# ─── Ordinales CONSUMIDORES ────────────────────────────────────────────────────

NSE_ORD = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}
NSE_ORD_INV = {v: k for k, v in NSE_ORD.items()}

CONSUMIDORES_EDU_ORD = {
    "secundaria_incompleta": 0,
    "secundaria_completa": 1,
    "tecnica_incompleta": 2,
    "tecnica_completa": 3,
    "universitaria_incompleta": 4,
    "universitaria_completa": 5,
    "posgrado": 6,
}
CONSUMIDORES_EDU_ORD_INV = {v: k for k, v in CONSUMIDORES_EDU_ORD.items()}

_IC_VARS_CONSUMIDORES = [
    "edad_num",          # 0
    "nse_num",           # 1 (E=0..A=4)
    "nivel_edu_num",     # 2 (0..6)
    "banca_digital_num", # 3 (binary)
    "ingreso_num",       # 4
    "region_lima_num",   # 5 (binary)
    "compra_online_num", # 6 (binary)
]

# ─── Ordinales FINANCIERO ──────────────────────────────────────────────────────

BANCARIZACION_ORD = {"basico": 0, "intermedio": 1, "avanzado": 2}
BANCARIZACION_ORD_INV = {v: k for k, v in BANCARIZACION_ORD.items()}

FINANCIERO_EDU_ORD = {
    "primaria_o_menos": 0,
    "secundaria_incompleta": 1,
    "secundaria_completa": 2,
    "tecnica": 3,
    "universitaria_incompleta": 4,
    "universitaria_completa": 5,
    "posgrado": 6,
}
FINANCIERO_EDU_ORD_INV = {v: k for k, v in FINANCIERO_EDU_ORD.items()}

_IC_VARS_FINANCIERO = [
    "edad_num",          # 0
    "nivel_edu_num",     # 1 (0..6)
    "bancarizacion_num", # 2 (0..2)
    "credito_num",       # 3 (binary)
    "canal_digital_num", # 4 (binary: app_movil/web)
    "ingreso_num",       # 5
    "yape_num",          # 6 (binary)
    "region_lima_num",   # 7 (binary)
]

# ─── Ordinales SALUD ──────────────────────────────────────────────────────────

SEGURO_ORD = {
    "sin_seguro": 0,
    "sis_gratuito": 1,
    "sis_independiente_pagante": 2,
    "essalud": 3,
    "ffaa_policia": 4,
    "seguro_privado_eps": 5,
}
SEGURO_ORD_INV = {v: k for k, v in SEGURO_ORD.items()}

SALUD_EDU_ORD = {
    "sin_nivel": 0,
    "primaria": 1,
    "secundaria_incompleta": 2,
    "secundaria_completa": 3,
    "tecnica": 4,
    "universitaria": 5,
    "posgrado": 6,
}
SALUD_EDU_ORD_INV = {v: k for k, v in SALUD_EDU_ORD.items()}

_IC_VARS_SALUD = [
    "edad_num",       # 0
    "nivel_edu_num",  # 1 (0..6)
    "seguro_num",     # 2 (0..5)
    "zona_rural_num", # 3 (binary)
    "ingreso_num",    # 4
    "quechua_num",    # 5 (binary: quechua/aymara)
    "cronica_num",    # 6 (binary)
    "region_lima_num",# 7 (binary)
]

# ─── Configuración de columnas SDV por segmento ────────────────────────────────

_SEGMENT_COLUMN_TYPES: dict[str, dict[str, list[str]]] = {
    "mype": {
        "categorical": [
            "region", "rubro", "tamaño", "formalizado", "canal_venta",
            "adopcion_digital", "credito", "nivel_educativo", "lengua_materna",
        ],
        "numerical": ["edad_dueño", "ingreso_mensual_soles"],
    },
    "consumidores": {
        "categorical": [
            "region", "nivel_socioeconomico", "nivel_educativo",
            "dispositivo_principal", "marketplace_preferido",
            "metodo_pago_preferido", "lengua_materna", "ocupacion",
            "sexo", "frecuencia_uso_internet",
        ],
        "numerical": ["edad", "ingreso_mensual_soles"],
    },
    "financiero": {
        "categorical": [
            "region", "tipo_entidad_principal", "nivel_bancarizacion",
            "canal_preferido", "nivel_educativo", "lengua_materna",
            "ocupacion", "sexo",
        ],
        "numerical": ["edad", "ingreso_mensual_soles"],
    },
    "salud": {
        "categorical": [
            "region", "tipo_seguro", "zona", "establecimiento_preferido",
            "motivo_consulta_principal", "nivel_educativo", "lengua_materna",
            "sexo",
        ],
        "numerical": ["edad", "ingreso_mensual_soles"],
    },
}


class Calibrator:
    """
    Carga distribuciones fuente JSON → genera dataset de entrenamiento calibrado
    → fitea GaussianCopulaSynthesizer → expone sample(n).
    """

    def __init__(self, segmento: str, fuente_dir: str = "data/fuentes") -> None:
        self.segmento = segmento
        self.fuente_path = Path(fuente_dir)
        self._fuente: dict | None = None
        self._synthesizer = None
        self._cache_path = Path(f".sdv_model_{segmento}.pkl")

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, n_training: int = 5000, use_cache: bool = True) -> None:
        """Fitea el sintetizador. Usa caché si existe para evitar re-entrenamiento."""
        if use_cache and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._synthesizer = pickle.load(f)  # nosec: caché local
            return

        import warnings

        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        df = self._build_training_data(n_training)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        col_types = _SEGMENT_COLUMN_TYPES.get(self.segmento, _SEGMENT_COLUMN_TYPES["mype"])
        for col in col_types["categorical"]:
            if col in df.columns:
                metadata.update_column(column_name=col, sdtype="categorical")
        for col in col_types["numerical"]:
            if col in df.columns:
                metadata.update_column(column_name=col, sdtype="numerical")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            synth = GaussianCopulaSynthesizer(metadata)
            synth.fit(df)
        self._synthesizer = synth

        with open(self._cache_path, "wb") as f:
            pickle.dump(synth, f)  # nosec: caché local

    def sample(self, n: int) -> pd.DataFrame:
        """Genera n filas sintéticas. Requiere fit() previo."""
        if self._synthesizer is None:
            raise RuntimeError("Llama a fit() antes de sample().")
        return self._synthesizer.sample(num_rows=n)

    # ------------------------------------------------------------------ #
    # Despacho por segmento                                                #
    # ------------------------------------------------------------------ #

    def _build_training_data(self, n: int = 5000) -> pd.DataFrame:
        if self.segmento == "mype":
            return self._build_training_data_mype(n)
        elif self.segmento == "consumidores":
            return self._build_training_data_consumidores(n)
        elif self.segmento == "financiero":
            return self._build_training_data_financiero(n)
        elif self.segmento == "salud":
            return self._build_training_data_salud(n)
        raise ValueError(
            f"Segmento '{self.segmento}' sin implementación de datos de entrenamiento."
        )

    # ------------------------------------------------------------------ #
    # Utilidades compartidas                                               #
    # ------------------------------------------------------------------ #

    def _load_fuente(self) -> dict:
        if self._fuente is None:
            fname = SEGMENTO_FUENTE.get(self.segmento)
            if fname is None:
                raise ValueError(
                    f"Segmento '{self.segmento}' sin fuente configurada. "
                    f"Disponibles: {list(SEGMENTO_FUENTE)}"
                )
            path = self.fuente_path / fname
            if not path.exists():
                raise FileNotFoundError(f"Archivo de fuente no encontrado: {path}")
            with open(path, encoding="utf-8") as f:
                self._fuente = json.load(f)
        return self._fuente

    @staticmethod
    def _nearest_psd(C: np.ndarray) -> np.ndarray:
        """Proyecta C a la correlación semi-definida positiva más cercana (Higham 2002)."""
        eigvals, eigvecs = la.eigh(C)
        eigvals = np.maximum(eigvals, 1e-8)
        C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(C_psd))
        return C_psd / np.outer(d, d)

    @staticmethod
    def _sample_probs(valores: dict, n: int, rng: np.random.Generator) -> np.ndarray:
        keys = list(valores.keys())
        probs = np.array(list(valores.values()), dtype=float)
        return rng.choice(keys, size=n, p=probs / probs.sum())

    @staticmethod
    def _iman_conover(
        cols: dict[str, np.ndarray],
        col_order: list[str],
        C: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """
        Método Iman-Conover: reordena cada columna para inducir correlaciones de rango
        preservando exactamente las distribuciones marginales.
        """
        n = len(next(iter(cols.values())))
        L = la.cholesky(C + 1e-10 * np.eye(len(col_order)))
        z = rng.standard_normal((n, len(col_order)))
        normal_samples = z @ L.T

        result = {col: arr.copy() for col, arr in cols.items()}
        for i, col in enumerate(col_order):
            if col not in cols:
                continue
            target_ranks = np.argsort(np.argsort(normal_samples[:, i]))
            result[col] = np.sort(cols[col])[target_ranks]

        return result

    @staticmethod
    def _propagate_region_lima(
        induced_region_lima: np.ndarray,
        region_raw: np.ndarray,
        region_valores: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Propaga cambios de region_lima_num inducido al campo categórico region."""
        region_final = region_raw.copy()
        non_lima = [r for r in region_valores if r != "lima_metropolitana"]
        non_lima_p = np.array([region_valores[r] for r in non_lima])
        non_lima_p /= non_lima_p.sum()

        became_lima = (induced_region_lima >= 0.5) & (region_raw != "lima_metropolitana")
        became_non_lima = (induced_region_lima < 0.5) & (region_raw == "lima_metropolitana")
        if became_lima.any():
            region_final[became_lima] = "lima_metropolitana"
        if became_non_lima.any():
            region_final[became_non_lima] = rng.choice(
                non_lima, size=int(became_non_lima.sum()), p=non_lima_p
            )
        return region_final

    @staticmethod
    def _lognormal_sample(info: dict, n: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea una distribución log-normal desde media/std del JSON fuente."""
        sigma_ln = np.sqrt(np.log(1 + (info["std"] / info["media"]) ** 2))
        mu_ln = np.log(info["media"]) - sigma_ln**2 / 2
        return np.clip(rng.lognormal(mu_ln, sigma_ln, n), info["min"], info["max"])

    # ------------------------------------------------------------------ #
    # MYPE                                                                 #
    # ------------------------------------------------------------------ #

    def _build_corr_matrix_mype(self, corrs: dict) -> np.ndarray:
        """
        Construye la matriz de correlación de rango 9×9 para el segmento MYPE.

        Variables (índices según _IC_VARS):
          0=edad  1=nivel_edu  2=adopcion  3=formalizado  4=credito_formal
          5=tamaño  6=ingreso  7=region_lima  8=canal_whatsapp
        """
        C = np.eye(9)

        def s(i: int, j: int, v: float) -> None:
            C[i, j] = v
            C[j, i] = v

        s(0, 2, corrs.get("adopcion_digital_vs_edad", -0.31))
        s(1, 2, corrs.get("adopcion_digital_vs_nivel_educativo", 0.42))
        s(3, 4, corrs.get("credito_formal_vs_formalizado", 0.55))
        s(5, 6, corrs.get("ingreso_vs_tamaño", 0.48))
        s(7, 2, corrs.get("adopcion_digital_vs_region_lima", 0.38))
        s(0, 8, -corrs.get("canal_whatsapp_vs_edad_menor_45", 0.29))

        # Correlaciones secundarias: inferidas de dominio (*)
        s(1, 3, 0.20)
        s(2, 8, 0.30)
        s(7, 1, 0.15)

        return self._nearest_psd(C)

    def _build_training_data_mype(self, n: int = 5000) -> pd.DataFrame:
        f = self._load_fuente()
        corrs = f.get("correlaciones_conocidas", {})
        rng = np.random.default_rng(42)

        region_raw = self._sample_probs(f["region"]["valores"], n, rng)
        rubro_raw = self._sample_probs(f["rubro"]["valores"], n, rng)

        tamaño_raw = self._sample_probs(f["tamaño"]["valores"], n, rng)
        tamaño_num = np.array([TAMAÑO_ORD[v] for v in tamaño_raw], dtype=float)

        p_formal = f["formalizado"]["con_ruc"]
        formalizado_num = rng.binomial(1, p_formal, n).astype(float)

        canal = f["canal_venta"]
        canal_whatsapp_num = rng.binomial(1, canal["whatsapp_incluido"], n).astype(float)

        ad = f["adopcion_digital"]
        adopcion_raw = self._sample_probs(
            {"nula": ad["nula"], "baja": ad["baja"], "media": ad["media"], "alta": ad["alta"]},
            n, rng,
        )
        adopcion_num = np.array([ADOPCION_ORD[v] for v in adopcion_raw], dtype=float)

        cred = f["credito"]
        p_cred_formal = (
            cred["credito_formal_banco"]
            + cred["credito_formal_caja_municipal"]
            + cred["credito_formal_financiera"]
        )
        credito_formal_num = rng.binomial(1, p_cred_formal, n).astype(float)

        ed = f["edad_dueño"]
        edad_num = np.clip(rng.normal(ed["media"], ed["std"], n), ed["min"], ed["max"])

        nivel_raw = self._sample_probs(f["nivel_educativo"]["valores"], n, rng)
        nivel_edu_num = np.array([EDUCACION_ORD[v] for v in nivel_raw], dtype=float)

        ingreso_num = self._lognormal_sample(f["ingreso_mensual_soles"], n, rng)
        region_lima_num = np.where(region_raw == "lima_metropolitana", 1.0, 0.0)
        lengua_raw = self._sample_probs(f["lengua_materna"]["valores"], n, rng)

        ic_input: dict[str, np.ndarray] = {
            "edad_num": edad_num,
            "nivel_edu_num": nivel_edu_num,
            "adopcion_num": adopcion_num,
            "formalizado_num": formalizado_num,
            "credito_formal_num": credito_formal_num,
            "tamaño_num": tamaño_num,
            "ingreso_num": ingreso_num,
            "region_lima_num": region_lima_num,
            "canal_whatsapp_num": canal_whatsapp_num,
        }
        C = self._build_corr_matrix_mype(corrs)
        induced = self._iman_conover(ic_input, _IC_VARS, C, rng)

        def to_adopcion(arr: np.ndarray) -> np.ndarray:
            return np.array([ADOPCION_ORD_INV[int(np.clip(round(v), 0, 3))] for v in arr])

        def to_tamaño(arr: np.ndarray) -> np.ndarray:
            return np.array([TAMAÑO_ORD_INV[int(np.clip(round(v), 0, 3))] for v in arr])

        def to_nivel(arr: np.ndarray) -> np.ndarray:
            return np.array([EDUCACION_ORD_INV[int(np.clip(round(v), 0, 8))] for v in arr])

        region_final = self._propagate_region_lima(
            induced["region_lima_num"], region_raw, f["region"]["valores"], rng
        )

        p_nf = 1.0 - p_cred_formal
        formal_choices = rng.choice(
            ["credito_formal_banco", "credito_formal_caja_municipal", "credito_formal_financiera"],
            size=n,
            p=[
                cred["credito_formal_banco"] / p_cred_formal,
                cred["credito_formal_caja_municipal"] / p_cred_formal,
                cred["credito_formal_financiera"] / p_cred_formal,
            ],
        )
        informal_choices = rng.choice(
            ["sin_credito", "credito_informal_prestamista", "credito_informal_familiar"],
            size=n,
            p=[
                cred["sin_credito"] / p_nf,
                cred["credito_informal_prestamista"] / p_nf,
                cred["credito_informal_familiar"] / p_nf,
            ],
        )
        credito_final = np.where(
            induced["credito_formal_num"] >= 0.5, formal_choices, informal_choices
        )

        return pd.DataFrame({
            "region": region_final,
            "rubro": rubro_raw,
            "tamaño": to_tamaño(induced["tamaño_num"]),
            "formalizado": np.where(
                induced["formalizado_num"] >= 0.5, "con_ruc", "completamente_informal"
            ),
            "canal_venta": np.where(
                induced["canal_whatsapp_num"] >= 0.5, "whatsapp_incluido", "fisico_exclusivo"
            ),
            "adopcion_digital": to_adopcion(induced["adopcion_num"]),
            "credito": credito_final,
            "edad_dueño": induced["edad_num"].round(1),
            "nivel_educativo": to_nivel(induced["nivel_edu_num"]),
            "ingreso_mensual_soles": induced["ingreso_num"].round(0).astype(int),
            "lengua_materna": lengua_raw,
        })

    # ------------------------------------------------------------------ #
    # CONSUMIDORES                                                         #
    # ------------------------------------------------------------------ #

    def _build_corr_matrix_consumidores(self, corrs: dict) -> np.ndarray:
        """
        Matriz de correlación 7×7 para consumidores digitales.

        Variables (índices según _IC_VARS_CONSUMIDORES):
          0=edad  1=nse  2=nivel_edu  3=banca_digital  4=ingreso
          5=region_lima  6=compra_online
        """
        C = np.eye(7)

        def s(i: int, j: int, v: float) -> None:
            C[i, j] = v
            C[j, i] = v

        # Correlaciones documentadas en JSON fuente
        s(3, 0, corrs.get("banca_digital_vs_edad_inversa", -0.29))
        s(1, 6, corrs.get("compra_online_vs_nse", 0.44))
        s(2, 6, corrs.get("compra_online_vs_nivel_educativo", 0.38))
        s(4, 6, corrs.get("ingreso_vs_ticket_compra", 0.41))

        # Correlaciones secundarias de dominio (*)
        s(1, 2, 0.40)   # NSE ↔ nivel_edu
        s(5, 1, 0.25)   # region_lima ↔ NSE
        s(5, 2, 0.15)   # region_lima ↔ nivel_edu
        s(1, 4, 0.35)   # NSE ↔ ingreso
        s(3, 6, 0.30)   # banca_digital ↔ compra_online

        return self._nearest_psd(C)

    def _build_training_data_consumidores(self, n: int = 5000) -> pd.DataFrame:
        f = self._load_fuente()
        corrs = f.get("correlaciones_conocidas", {})
        rng = np.random.default_rng(42)

        # ── Marginals ─────────────────────────────────────────────────────
        region_raw = self._sample_probs(f["region"]["valores"], n, rng)
        nse_raw = self._sample_probs(f["nivel_socioeconomico"]["valores"], n, rng)
        nse_num = np.array([NSE_ORD[v] for v in nse_raw], dtype=float)

        nivel_raw = self._sample_probs(f["nivel_educativo"]["valores"], n, rng)
        nivel_edu_num = np.array([CONSUMIDORES_EDU_ORD[v] for v in nivel_raw], dtype=float)

        ed = f["edad"]
        edad_num = np.clip(rng.normal(ed["media"], ed["std"], n), ed["min"], ed["max"])

        banca_digital_num = rng.binomial(1, f["banca_digital"]["usa_app_banco"], n).astype(float)
        compra_online_num = rng.binomial(1, f["ecommerce"]["compro_ultimo_mes"], n).astype(float)
        ingreso_num = self._lognormal_sample(f["ingreso_mensual_soles"], n, rng)
        region_lima_num = np.where(region_raw == "lima_metropolitana", 1.0, 0.0)

        # Marginals sin IC (no participan en correlaciones documentadas)
        dispositivo_raw = self._sample_probs(f["dispositivo_principal"]["valores"], n, rng)
        marketplace_raw = self._sample_probs(f["ecommerce"]["marketplace_preferido"], n, rng)
        metodo_pago_raw = self._sample_probs(f["ecommerce"]["metodo_pago_preferido"], n, rng)
        lengua_raw = self._sample_probs(f["lengua_materna"]["valores"], n, rng)
        ocupacion_raw = self._sample_probs(f["ocupacion"]["valores"], n, rng)
        frecuencia_raw = self._sample_probs(f["frecuencia_uso_internet"]["valores"], n, rng)
        sexo_raw = rng.choice(
            ["masculino", "femenino"], size=n,
            p=[f["sexo"]["masculino"], f["sexo"]["femenino"]],
        )

        # ── Inducir correlaciones (Iman-Conover) ──────────────────────────
        ic_input: dict[str, np.ndarray] = {
            "edad_num": edad_num,
            "nse_num": nse_num,
            "nivel_edu_num": nivel_edu_num,
            "banca_digital_num": banca_digital_num,
            "ingreso_num": ingreso_num,
            "region_lima_num": region_lima_num,
            "compra_online_num": compra_online_num,
        }
        C = self._build_corr_matrix_consumidores(corrs)
        induced = self._iman_conover(ic_input, _IC_VARS_CONSUMIDORES, C, rng)

        # ── Mapear numéricos → categorías ─────────────────────────────────
        def to_nse(arr: np.ndarray) -> np.ndarray:
            return np.array([NSE_ORD_INV[int(np.clip(round(v), 0, 4))] for v in arr])

        def to_nivel_cons(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [CONSUMIDORES_EDU_ORD_INV[int(np.clip(round(v), 0, 6))] for v in arr]
            )

        region_final = self._propagate_region_lima(
            induced["region_lima_num"], region_raw, f["region"]["valores"], rng
        )

        return pd.DataFrame({
            "region": region_final,
            "nivel_socioeconomico": to_nse(induced["nse_num"]),
            "nivel_educativo": to_nivel_cons(induced["nivel_edu_num"]),
            "dispositivo_principal": dispositivo_raw,
            "marketplace_preferido": marketplace_raw,
            "metodo_pago_preferido": metodo_pago_raw,
            "lengua_materna": lengua_raw,
            "ocupacion": ocupacion_raw,
            "sexo": sexo_raw,
            "frecuencia_uso_internet": frecuencia_raw,
            "edad": induced["edad_num"].round(1),
            "ingreso_mensual_soles": induced["ingreso_num"].round(0).astype(int),
        })

    # ------------------------------------------------------------------ #
    # FINANCIERO                                                           #
    # ------------------------------------------------------------------ #

    def _build_corr_matrix_financiero(self, corrs: dict) -> np.ndarray:
        """
        Matriz de correlación 8×8 para usuarios del sistema financiero.

        Variables (índices según _IC_VARS_FINANCIERO):
          0=edad  1=nivel_edu  2=bancarizacion  3=credito
          4=canal_digital  5=ingreso  6=yape  7=region_lima
        """
        C = np.eye(8)

        def s(i: int, j: int, v: float) -> None:
            C[i, j] = v
            C[j, i] = v

        # Correlaciones documentadas en JSON fuente
        s(2, 5, corrs.get("nivel_bancarizacion_vs_ingreso", 0.52))
        s(3, 5, corrs.get("credito_activo_vs_ingreso", 0.41))
        s(4, 0, corrs.get("canal_digital_vs_edad_inversa", -0.38))
        s(4, 1, corrs.get("canal_digital_vs_nivel_educativo", 0.44))
        # tarjeta_credito_vs_nivel_bancarizacion → proxy: bancarizacion ↔ credito
        s(2, 3, corrs.get("tarjeta_credito_vs_nivel_bancarizacion", 0.58))
        s(1, 2, corrs.get("confianza_vs_nivel_educativo", 0.31))
        # yape_vs_edad_menor_40: positivo con ser joven → negativo con edad continua
        s(6, 0, -corrs.get("yape_vs_edad_menor_40", 0.42))

        # Correlaciones secundarias de dominio (*)
        s(7, 1, 0.15)   # region_lima ↔ nivel_edu
        s(7, 4, 0.20)   # region_lima ↔ canal_digital

        return self._nearest_psd(C)

    def _build_training_data_financiero(self, n: int = 5000) -> pd.DataFrame:
        f = self._load_fuente()
        corrs = f.get("correlaciones_conocidas", {})
        rng = np.random.default_rng(42)

        # ── Marginals ─────────────────────────────────────────────────────
        region_raw = self._sample_probs(f["region"]["valores"], n, rng)

        ed = f["edad"]
        edad_num = np.clip(rng.normal(ed["media"], ed["std"], n), ed["min"], ed["max"])

        nivel_raw = self._sample_probs(f["nivel_educativo"]["valores"], n, rng)
        nivel_edu_num = np.array([FINANCIERO_EDU_ORD[v] for v in nivel_raw], dtype=float)

        bancarizacion_raw = self._sample_probs(f["nivel_bancarizacion"]["valores"], n, rng)
        bancarizacion_num = np.array([BANCARIZACION_ORD[v] for v in bancarizacion_raw], dtype=float)

        credito_num = rng.binomial(1, f["credito"]["tiene_credito_activo"], n).astype(float)

        # canal_digital: app_movil o web_banca_online
        p_digital = f["canal_preferido"]["valores"]["app_movil"] + f["canal_preferido"]["valores"]["web_banca_online"]
        canal_digital_num = rng.binomial(1, p_digital, n).astype(float)

        ingreso_num = self._lognormal_sample(f["ingreso_mensual_soles"], n, rng)

        yape_num = rng.binomial(1, f["inclusion_digital_financiera"]["usa_yape"], n).astype(float)

        region_lima_num = np.where(region_raw == "lima_metropolitana", 1.0, 0.0)

        # Marginals sin IC
        tipo_entidad_raw = self._sample_probs(f["tipo_entidad_principal"]["valores"], n, rng)
        lengua_raw = self._sample_probs(f["lengua_materna"]["valores"], n, rng)
        ocupacion_raw = self._sample_probs(f["ocupacion"]["valores"], n, rng)
        sexo_raw = rng.choice(
            ["masculino", "femenino"], size=n,
            p=[f["sexo"]["masculino"], f["sexo"]["femenino"]],
        )

        # ── Inducir correlaciones (Iman-Conover) ──────────────────────────
        ic_input: dict[str, np.ndarray] = {
            "edad_num": edad_num,
            "nivel_edu_num": nivel_edu_num,
            "bancarizacion_num": bancarizacion_num,
            "credito_num": credito_num,
            "canal_digital_num": canal_digital_num,
            "ingreso_num": ingreso_num,
            "yape_num": yape_num,
            "region_lima_num": region_lima_num,
        }
        C = self._build_corr_matrix_financiero(corrs)
        induced = self._iman_conover(ic_input, _IC_VARS_FINANCIERO, C, rng)

        # ── Mapear numéricos → categorías ─────────────────────────────────
        def to_bancarizacion(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [BANCARIZACION_ORD_INV[int(np.clip(round(v), 0, 2))] for v in arr]
            )

        def to_nivel_fin(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [FINANCIERO_EDU_ORD_INV[int(np.clip(round(v), 0, 6))] for v in arr]
            )

        # canal_preferido: digital (app/web) vs no-digital (agente/ventanilla/cajero)
        canal_vals = f["canal_preferido"]["valores"]
        digital_keys = ["app_movil", "web_banca_online"]
        non_digital_keys = [k for k in canal_vals if k not in digital_keys]
        p_dig = np.array([canal_vals[k] for k in digital_keys])
        p_ndig = np.array([canal_vals[k] for k in non_digital_keys])
        digital_choices = rng.choice(digital_keys, size=n, p=p_dig / p_dig.sum())
        non_digital_choices = rng.choice(non_digital_keys, size=n, p=p_ndig / p_ndig.sum())
        canal_final = np.where(
            induced["canal_digital_num"] >= 0.5, digital_choices, non_digital_choices
        )

        region_final = self._propagate_region_lima(
            induced["region_lima_num"], region_raw, f["region"]["valores"], rng
        )

        return pd.DataFrame({
            "region": region_final,
            "tipo_entidad_principal": tipo_entidad_raw,
            "nivel_bancarizacion": to_bancarizacion(induced["bancarizacion_num"]),
            "canal_preferido": canal_final,
            "nivel_educativo": to_nivel_fin(induced["nivel_edu_num"]),
            "lengua_materna": lengua_raw,
            "ocupacion": ocupacion_raw,
            "sexo": sexo_raw,
            "edad": induced["edad_num"].round(1),
            "ingreso_mensual_soles": induced["ingreso_num"].round(0).astype(int),
        })

    # ------------------------------------------------------------------ #
    # SALUD                                                                #
    # ------------------------------------------------------------------ #

    def _build_corr_matrix_salud(self, corrs: dict) -> np.ndarray:
        """
        Matriz de correlación 8×8 para usuarios del sistema de salud.

        Variables (índices según _IC_VARS_SALUD):
          0=edad  1=nivel_edu  2=seguro  3=zona_rural
          4=ingreso  5=quechua  6=cronica  7=region_lima
        """
        C = np.eye(8)

        def s(i: int, j: int, v: float) -> None:
            C[i, j] = v
            C[j, i] = v

        # Correlaciones documentadas en JSON fuente
        s(6, 0, corrs.get("enfermedad_cronica_vs_edad", 0.48))
        # sis_vs_ingreso_bajo = 0.55 → tipo_seguro (más formal) → más ingreso
        s(2, 4, corrs.get("clinica_privada_vs_ingreso_alto", 0.49))
        s(3, 5, corrs.get("quechua_vs_zona_rural", 0.58))
        s(3, 6, corrs.get("anemia_vs_zona_rural", 0.34))

        # Correlaciones secundarias de dominio (*)
        s(1, 2, 0.30)    # nivel_edu ↔ tipo_seguro: más edu → mejor seguro
        s(5, 7, -0.35)   # quechua ↔ region_lima: Lima tiene menos quechua
        s(3, 7, -0.50)   # zona_rural ↔ region_lima: Lima es urbana
        s(0, 3, 0.15)    # edad ↔ zona_rural: zonas rurales son algo más envejecidas
        s(4, 1, 0.25)    # ingreso ↔ nivel_edu: dominio
        s(2, 7, 0.25)    # seguro ↔ region_lima: Lima tiene más seguro privado/essalud

        return self._nearest_psd(C)

    def _build_training_data_salud(self, n: int = 5000) -> pd.DataFrame:
        f = self._load_fuente()
        corrs = f.get("correlaciones_conocidas", {})
        rng = np.random.default_rng(42)

        # ── Marginals ─────────────────────────────────────────────────────
        region_raw = self._sample_probs(f["region"]["valores"], n, rng)

        ed = f["edad"]
        edad_num = np.clip(rng.normal(ed["media"], ed["std"], n), ed["min"], ed["max"])

        nivel_raw = self._sample_probs(f["nivel_educativo"]["valores"], n, rng)
        nivel_edu_num = np.array([SALUD_EDU_ORD[v] for v in nivel_raw], dtype=float)

        seguro_raw = self._sample_probs(f["tipo_seguro"]["valores"], n, rng)
        seguro_num = np.array([SEGURO_ORD[v] for v in seguro_raw], dtype=float)

        zona_rural_num = rng.binomial(1, f["zona"]["rural"], n).astype(float)

        ingreso_num = self._lognormal_sample(f["ingreso_mensual_soles"], n, rng)

        # quechua_num: lengua quechua, aymara o bilingüe_quechua
        lengua_vals = f["lengua_materna"]["valores"]
        p_quechua = sum(
            v for k, v in lengua_vals.items()
            if "quechua" in k or "aymara" in k
        )
        quechua_num = rng.binomial(1, p_quechua, n).astype(float)

        # cronica_num: prevalencia sin ninguna_cronica
        p_cronica = 1.0 - f["enfermedades_cronicas"]["ninguna_cronica"]
        cronica_num = rng.binomial(1, p_cronica, n).astype(float)

        region_lima_num = np.where(region_raw == "lima_metropolitana", 1.0, 0.0)

        # Marginals sin IC
        establecimiento_raw = self._sample_probs(f["establecimiento_preferido"]["valores"], n, rng)
        motivo_raw = self._sample_probs(f["motivo_consulta_principal"]["valores"], n, rng)
        sexo_raw = rng.choice(
            ["femenino", "masculino"], size=n,
            p=[f["sexo"]["femenino"], f["sexo"]["masculino"]],
        )

        # ── Inducir correlaciones (Iman-Conover) ──────────────────────────
        ic_input: dict[str, np.ndarray] = {
            "edad_num": edad_num,
            "nivel_edu_num": nivel_edu_num,
            "seguro_num": seguro_num,
            "zona_rural_num": zona_rural_num,
            "ingreso_num": ingreso_num,
            "quechua_num": quechua_num,
            "cronica_num": cronica_num,
            "region_lima_num": region_lima_num,
        }
        C = self._build_corr_matrix_salud(corrs)
        induced = self._iman_conover(ic_input, _IC_VARS_SALUD, C, rng)

        # ── Mapear numéricos → categorías ─────────────────────────────────
        def to_seguro(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [SEGURO_ORD_INV[int(np.clip(round(v), 0, 5))] for v in arr]
            )

        def to_nivel_salud(arr: np.ndarray) -> np.ndarray:
            return np.array(
                [SALUD_EDU_ORD_INV[int(np.clip(round(v), 0, 6))] for v in arr]
            )

        # zona: zona_rural_num inducido → urbana/rural
        zona_final = np.where(induced["zona_rural_num"] >= 0.5, "rural", "urbana")

        # lengua_materna: quechua_num inducido → pick quechua/aymara o castellano
        lengua_raw = self._sample_probs(lengua_vals, n, rng)
        quechua_vals = {k: v for k, v in lengua_vals.items() if "quechua" in k or "aymara" in k}
        castellano_vals = {k: v for k, v in lengua_vals.items() if k not in quechua_vals}
        qv_p = np.array(list(quechua_vals.values()))
        cv_p = np.array(list(castellano_vals.values()))
        quechua_choices = rng.choice(list(quechua_vals.keys()), size=n, p=qv_p / qv_p.sum())
        castellano_choices = rng.choice(list(castellano_vals.keys()), size=n, p=cv_p / cv_p.sum())
        lengua_final = np.where(
            induced["quechua_num"] >= 0.5, quechua_choices, castellano_choices
        )

        region_final = self._propagate_region_lima(
            induced["region_lima_num"], region_raw, f["region"]["valores"], rng
        )

        return pd.DataFrame({
            "region": region_final,
            "tipo_seguro": to_seguro(induced["seguro_num"]),
            "zona": zona_final,
            "establecimiento_preferido": establecimiento_raw,
            "motivo_consulta_principal": motivo_raw,
            "nivel_educativo": to_nivel_salud(induced["nivel_edu_num"]),
            "lengua_materna": lengua_final,
            "sexo": sexo_raw,
            "edad": induced["edad_num"].round(1),
            "ingreso_mensual_soles": induced["ingreso_num"].round(0).astype(int),
        })
