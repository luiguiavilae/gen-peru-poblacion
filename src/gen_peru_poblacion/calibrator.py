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
}

# Codificaciones ordinales para variables que participan en inducción de correlaciones
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

# Orden de columnas en la matriz de correlación (NO cambiar sin actualizar _build_corr_matrix)
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

        for col in [
            "region", "rubro", "tamaño", "formalizado", "canal_venta",
            "adopcion_digital", "credito", "nivel_educativo", "lengua_materna",
        ]:
            metadata.update_column(column_name=col, sdtype="categorical")

        metadata.update_column(column_name="edad_dueño", sdtype="numerical")
        metadata.update_column(column_name="ingreso_mensual_soles", sdtype="numerical")

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
    # Construcción de datos de entrenamiento                               #
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

    def _build_corr_matrix(self, corrs: dict) -> np.ndarray:
        """
        Construye la matriz de correlación de rango 9×9 desde los coeficientes
        documentados en correlaciones_conocidas del JSON fuente.

        Variables (índices según _IC_VARS):
          0=edad  1=nivel_edu  2=adopcion  3=formalizado  4=credito_formal
          5=tamaño  6=ingreso  7=region_lima  8=canal_whatsapp

        Las correlaciones marcadas (*) son secundarias: inferidas de dominio para
        mejorar el condicionamiento de la matriz. No provienen de datos fuente.
        """
        C = np.eye(9)

        def s(i: int, j: int, v: float) -> None:
            C[i, j] = v
            C[j, i] = v

        # Correlaciones documentadas en JSON fuente
        s(0, 2, corrs.get("adopcion_digital_vs_edad", -0.31))
        s(1, 2, corrs.get("adopcion_digital_vs_nivel_educativo", 0.42))
        s(3, 4, corrs.get("credito_formal_vs_formalizado", 0.55))
        s(5, 6, corrs.get("ingreso_vs_tamaño", 0.48))
        s(7, 2, corrs.get("adopcion_digital_vs_region_lima", 0.38))
        # canal_whatsapp_vs_edad_menor_45 = 0.29: jóvenes (edad baja) → más WhatsApp
        # → correlación negativa con edad continua
        s(0, 8, -corrs.get("canal_whatsapp_vs_edad_menor_45", 0.29))

        # Correlaciones secundarias (*): inferidas de dominio
        s(1, 3, 0.20)  # nivel_edu ↔ formalizado: más educado → más formal (*)
        s(2, 8, 0.30)  # adopcion_digital ↔ canal_whatsapp: digital → WhatsApp (*)
        s(7, 1, 0.15)  # region_lima ↔ nivel_edu: Lima → mayor educación (*)

        return self._nearest_psd(C)

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

        Para cada variable: ordena los valores originales según el ranking de una
        muestra multivariada normal con matriz de correlación C.
        """
        n = len(next(iter(cols.values())))
        # +1e-10 en diagonal garantiza definición positiva para Cholesky
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

    def _build_training_data(self, n: int = 5000) -> pd.DataFrame:
        f = self._load_fuente()
        corrs = f.get("correlaciones_conocidas", {})
        rng = np.random.default_rng(42)

        # ── 1. Muestreo de distribuciones marginales ────────────────────

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

        ing = f["ingreso_mensual_soles"]
        sigma_ln = np.sqrt(np.log(1 + (ing["std"] / ing["media"]) ** 2))
        mu_ln = np.log(ing["media"]) - sigma_ln**2 / 2
        ingreso_num = np.clip(rng.lognormal(mu_ln, sigma_ln, n), ing["min"], ing["max"])

        region_lima_num = np.where(region_raw == "lima_metropolitana", 1.0, 0.0)
        lengua_raw = self._sample_probs(f["lengua_materna"]["valores"], n, rng)

        # ── 2. Inducir correlaciones documentadas (Iman-Conover) ─────────

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
        C = self._build_corr_matrix(corrs)
        induced = self._iman_conover(ic_input, _IC_VARS, C, rng)

        # ── 3. Mapear numéricos inducidos → strings categóricos ──────────

        def to_adopcion(arr: np.ndarray) -> np.ndarray:
            return np.array([ADOPCION_ORD_INV[int(np.clip(round(v), 0, 3))] for v in arr])

        def to_tamaño(arr: np.ndarray) -> np.ndarray:
            return np.array([TAMAÑO_ORD_INV[int(np.clip(round(v), 0, 3))] for v in arr])

        def to_nivel(arr: np.ndarray) -> np.ndarray:
            return np.array([EDUCACION_ORD_INV[int(np.clip(round(v), 0, 8))] for v in arr])

        # Propagar cambios de region_lima al campo region categórico
        region_final = region_raw.copy()
        non_lima = [r for r in f["region"]["valores"] if r != "lima_metropolitana"]
        non_lima_p = np.array([f["region"]["valores"][r] for r in non_lima])
        non_lima_p /= non_lima_p.sum()

        became_lima = (induced["region_lima_num"] >= 0.5) & (region_raw != "lima_metropolitana")
        became_non_lima = (induced["region_lima_num"] < 0.5) & (region_raw == "lima_metropolitana")
        if became_lima.any():
            region_final[became_lima] = "lima_metropolitana"
        if became_non_lima.any():
            region_final[became_non_lima] = rng.choice(
                non_lima, size=int(became_non_lima.sum()), p=non_lima_p
            )

        # Crédito: respetar split formal/informal desde credito_formal_num inducido
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
        credito_final = np.where(induced["credito_formal_num"] >= 0.5, formal_choices, informal_choices)

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
