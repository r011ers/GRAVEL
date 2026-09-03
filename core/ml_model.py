"""
╔══════════════════════════════════════════════════════════════╗
║              GRAVEL v12 ULTRA — ml_model.py                 ║
║        Advanced Ensemble + Calibration + Uncertainty        ║
╚══════════════════════════════════════════════════════════════╝
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .features import PlayerFeatures
from .first_throw import (
    apply_logodds_shift,
    calibrate_elo_with_throw,
    calibrate_first_throw,
    match_closeness,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger("gravel_ml")


# ============================================================
# MATCH PREDICTION
# ============================================================

@dataclass
class MatchPrediction:
    player_a: str
    player_b: str

    prob_a: float
    prob_b: float

    expected_180s_a: float
    expected_180s_b: float

    expected_avg_a: float
    expected_avg_b: float

    model_version: str = "v13_ultra"

    raw_prob: float = 0.0
    elo_prob: float = 0.0
    edge_vs_elo: float = 0.0

    confidence: float = 0.0
    trust_ml: float = 0.0
    model_disagreement: float = 0.0
    uncertainty: float = 0.0
    edge_quality: str = "MEDIUM"

    # Calibración empírica first throw (MSS Bo7)
    first_throw_edge_pp: float = 0.0
    first_throw_closeness: float = 0.0
    throw_neutral_prob: float = 0.0



# ============================================================
# ADVANCED MODEL
# ============================================================

class AdvancedDartsModel:

    def __init__(self, db=None):

        self.db = db

        self.v8_model: Optional[Any] = None
        self.v8_scaler = None

        self.meta_scaler = None
        self.meta_calibrator = None

        self.load_error: Optional[str] = None

        self.model_dir = Path(__file__).parent.parent / "models"

        self._load_models()

    # ============================================================
    # LOAD MODELS
    # ============================================================

    def _load_models(self):
        self.load_error = None
        possible_paths = [
            self.model_dir / "darts_v8.pkl",
            Path(__file__).resolve().parent.parent / "models" / "darts_v8.pkl",
        ]

        target_path = None
        for p in possible_paths:
            if p.exists():
                target_path = p
                break

        if target_path is None:
            self.load_error = f"Model file 'darts_v8.pkl' not found in any expected location: {[str(p) for p in possible_paths]}"
            logger.error(self.load_error)
            return

        digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
        hash_file = target_path.with_name(target_path.name + ".sha256")
        if hash_file.is_file():
            expected = hash_file.read_text(encoding="utf-8").strip().split()[0]
            if digest != expected:
                self.load_error = (
                    f"SHA256 mismatch for {target_path.name}: "
                    f"got {digest}, expected {expected}"
                )
                logger.error(self.load_error)
                return
        else:
            logger.warning("Model hash file not found: %s", hash_file)

        try:
            with open(target_path, "rb") as f:
                loaded = pickle.load(f)

            if isinstance(loaded, dict):
                self.v8_model = loaded.get("model")
                self.v8_scaler = loaded.get("scaler")
                self.meta_scaler = loaded.get("meta_scaler")
                self.meta_calibrator = loaded.get("meta_calibrator")

                metrics = loaded.get("metrics", {})
                logger.info(
                    f"Loaded V13 bundle from {target_path.name} | "
                    f"Brier={metrics.get('cal_brier', 'N/A')}"
                )
            else:
                self.v8_model = loaded
                logger.info(f"Loaded legacy model from {target_path.name}.")

        except Exception as e:
            self.load_error = f"Error loading model from {target_path}: {e}"
            logger.error(self.load_error, exc_info=True)

    # ============================================================
    # ELO PROBABILITY
    # ============================================================

    def _get_elo_prob(self, elo_a, elo_b):

        return 1 / (
            1 + 10 ** ((elo_b - elo_a) / 400)
        )

    # ============================================================
    # META CALIBRATION
    # ============================================================

    def _apply_meta_calibration(
        self,
        raw_prob,
        elo_prob,
        feat_a,
        feat_b
    ):

        if self.meta_calibrator is None or self.meta_scaler is None:
            return raw_prob

        elo_gap = abs(
            feat_a.elo_rating -
            feat_b.elo_rating
        )

        model_features = np.array([[
            raw_prob,
            elo_prob,
            raw_prob - elo_prob,
            elo_gap,
            feat_a.form_wr_ewm,
            feat_b.form_wr_ewm
        ]])

        model_features_sc = self.meta_scaler.transform(model_features)

        calibrated = (
            self.meta_calibrator
            .predict_proba(model_features_sc)[0][1]
        )

        return float(
            np.clip(calibrated, 0.01, 0.99)
        )

    # ============================================================
    # CONFIDENCE SCORE
    # ============================================================

    def _confidence_score(
        self,
        prob,
        feat_a,
        feat_b
    ):

        edge = abs(prob - 0.5)

        matches_factor = min(min(feat_a.n_matches, feat_b.n_matches) / 50.0, 1.0)

        elo_gap = (
            abs(
                feat_a.elo_rating -
                feat_b.elo_rating
            ) / 400
        )

        confidence = (
            edge * 0.50 +
            matches_factor * 0.30 +
            elo_gap * 0.20
        )

        return float(np.clip(confidence, 0, 1))

    # ============================================================
    # FEATURE VECTOR
    # ============================================================

    def _build_feature_vector(
        self,
        feat_a,
        feat_b,
        h2h_a_winrate,
        first_throw_a
    ):

        fatigue_a = 1 if feat_a.session_position >= 4 else 0
        fatigue_b = 1 if feat_b.session_position >= 4 else 0

        momentum_avg_a = (
            feat_a.form_avg_ewm -
            feat_a.avg_score
        )

        momentum_avg_b = (
            feat_b.form_avg_ewm -
            feat_b.avg_score
        )

        elo_momentum_a = getattr(feat_a, "elo_momentum", 0.0)
        elo_momentum_b = getattr(feat_b, "elo_momentum", 0.0)


        return [

            feat_a.avg_score,
            feat_a.avg_checkout,
            feat_a.win_rate,
            feat_a.legs_win_rate,

            feat_a.form_avg_ewm,
            feat_a.form_checkout_ewm,
            feat_a.form_wr_ewm,
            feat_a.avg_slope,

            feat_b.avg_score,
            feat_b.avg_checkout,
            feat_b.win_rate,
            feat_b.legs_win_rate,

            feat_b.form_avg_ewm,
            feat_b.form_checkout_ewm,
            feat_b.form_wr_ewm,
            feat_b.avg_slope,

            h2h_a_winrate,

            1 if first_throw_a else 0,

            feat_a.pressure_co,
            feat_a.pressure_delta,
            feat_a.vs_similar_wr,

            feat_b.pressure_co,
            feat_b.pressure_delta,
            feat_b.vs_similar_wr,

            feat_a.days_since_last
            if feat_a.days_since_last is not None else 30,

            feat_b.days_since_last
            if feat_b.days_since_last is not None else 30,

            feat_a.elo_rating,
            feat_b.elo_rating,

            feat_a.elo_rating - feat_b.elo_rating,

            momentum_avg_a,
            momentum_avg_b,

            fatigue_a,
            fatigue_b,

            elo_momentum_a,
            elo_momentum_b,

            getattr(feat_a, "avg_std", 0),
            getattr(feat_b, "avg_std", 0),
        ]

    # ============================================================
    # CORE INFERENCE (una orientación de saque)
    # ============================================================

    def _infer_orientation(
        self,
        feat_a,
        feat_b,
        h2h_a_winrate: float,
        first_throw_a: bool,
        elo_prob_a: float,
    ):
        """
        Inferencia simétrica A↔B para una asignación fija de first throw.
        Devuelve (raw_prob_a, meta_prob_a).
        """
        # Forward (A vs B)
        X_full_f = self._build_feature_vector(
            feat_a, feat_b, h2h_a_winrate, first_throw_a
        )
        if hasattr(self.v8_model, "n_features_in_"):
            n_in = self.v8_model.n_features_in_
            X_f = np.array([X_full_f[:n_in]])
        else:
            n_in = None
            X_f = np.array([X_full_f])

        if self.v8_scaler is not None:
            X_f = self.v8_scaler.transform(X_f)

        raw_prob_a_f = float(self.v8_model.predict_proba(X_f)[0][1])

        # Backward (B vs A) con saque invertido
        X_full_b = self._build_feature_vector(
            feat_b, feat_a, 1.0 - h2h_a_winrate, not first_throw_a
        )
        if n_in is not None:
            X_b = np.array([X_full_b[:n_in]])
        else:
            X_b = np.array([X_full_b])

        if self.v8_scaler is not None:
            X_b = self.v8_scaler.transform(X_b)

        raw_prob_b_b = float(self.v8_model.predict_proba(X_b)[0][1])
        raw_prob_a = (raw_prob_a_f + (1.0 - raw_prob_b_b)) / 2.0

        meta_prob_f = self._apply_meta_calibration(
            raw_prob_a, elo_prob_a, feat_a, feat_b
        )
        meta_prob_b = self._apply_meta_calibration(
            1.0 - raw_prob_a, 1.0 - elo_prob_a, feat_b, feat_a
        )
        meta_prob = (meta_prob_f + (1.0 - meta_prob_b)) / 2.0
        return float(raw_prob_a), float(meta_prob)

    # ============================================================
    # MAIN PREDICTION
    # ============================================================

    def predict_match(
        self,
        feat_a,
        feat_b,
        h2h_a_winrate=0.5,
        first_throw_a=True
    ):

        if self.v8_model is None:
            self._load_models()

        if self.v8_model is None:
            err_details = self.load_error or "darts_v8.pkl not found"
            raise RuntimeError(
                f"ML model not loaded. Error detail: {err_details}"
            )

        # --- PROBABILIDAD DE ELO (throw-neutral) ---
        elo_prob_raw = self._get_elo_prob(
            feat_a.elo_rating,
            feat_b.elo_rating
        )

        # =========================================================
        # Doble orientación de saque → neutralizar sesgo ML
        # y reaplicar prior empírico MSS Bo7 (first_throw.py)
        # =========================================================
        raw_ft_a, meta_ft_a = self._infer_orientation(
            feat_a, feat_b, h2h_a_winrate, True, elo_prob_raw
        )
        raw_ft_b, meta_ft_b = self._infer_orientation(
            feat_a, feat_b, h2h_a_winrate, False, elo_prob_raw
        )

        # Raw simétrico con el saque real (para debug/metadata)
        raw_prob_a = raw_ft_a if first_throw_a else raw_ft_b

        avg_a = float(getattr(feat_a, "form_avg_ewm", None) or feat_a.avg_score or 85.0)
        avg_b = float(getattr(feat_b, "form_avg_ewm", None) or feat_b.avg_score or 85.0)

        ft_adj = calibrate_first_throw(
            prob_if_a_throws=meta_ft_a,
            prob_if_b_throws=meta_ft_b,
            first_throw_a=bool(first_throw_a),
            elo_a=float(feat_a.elo_rating),
            elo_b=float(feat_b.elo_rating),
            avg_a=avg_a,
            avg_b=avg_b,
            wr_a=float(feat_a.win_rate),
            wr_b=float(feat_b.win_rate),
        )

        # ELO también recibe el prior de saque empírico
        elo_prob_a = calibrate_elo_with_throw(
            elo_prob_raw,
            first_throw_a=bool(first_throw_a),
            closeness=ft_adj.closeness,
        )

        meta_prob = float(ft_adj.prob)

        # Ponderaciones y factores de experiencia
        matches_a = max(feat_a.n_matches, 1)
        matches_b = max(feat_b.n_matches, 1)
        experience_factor = min(matches_a, matches_b)

        elo_gap = abs(
            feat_a.elo_rating -
            feat_b.elo_rating
        )

        # =========================================================
        # v14 CALIBRATION BLEND (mejora probs sin reentrenar)
        # 1) Ensemble ML↔ELO cuando hay desacuerdo
        # 2) Tilt suave por forma live (L3 / ewm)
        # 3) Shrinkage REAL hacia ancla (antes solo se guardaba)
        # =========================================================

        model_disagreement = abs(meta_prob - elo_prob_a)

        # (1) Si ML y ELO divergen, tirar hacia ELO (más estable en EV)
        #     Más peso a ELO con poca muestra o desacuerdo alto.
        elo_blend = 0.0
        if model_disagreement > 0.06:
            elo_blend = min(0.38, (model_disagreement - 0.06) * 1.35)
        if experience_factor < 20:
            elo_blend = min(0.48, elo_blend + 0.08 * (1.0 - experience_factor / 20.0))
        if elo_blend > 0:
            meta_prob = (1.0 - elo_blend) * meta_prob + elo_blend * elo_prob_a

        # (2) Forma reciente: pequeño shift en log-odds (máx ~±3.5 pp)
        form_a = float(getattr(feat_a, "form_wr_ewm", None) or getattr(feat_a, "form_win_rate", 0.5) or 0.5)
        form_b = float(getattr(feat_b, "form_wr_ewm", None) or getattr(feat_b, "form_win_rate", 0.5) or 0.5)
        form_gap = float(np.clip(form_a - form_b, -0.55, 0.55))
        # Confianza de forma: más partidos + actividad reciente
        form_n = min(matches_a, matches_b, 30) / 30.0
        form_delta = 0.18 * form_gap * (0.45 + 0.55 * form_n)
        if abs(form_delta) > 1e-6:
            meta_prob = apply_logodds_shift(meta_prob, form_delta)

        # Recalcular desacuerdo post-blend (metadata honesta)
        model_disagreement = abs(meta_prob - elo_prob_a)

        recent_weight_a = np.exp(-((feat_a.days_since_last or 30) / 7.0))
        recent_weight_b = np.exp(-((feat_b.days_since_last or 30) / 7.0))
        recency_factor = (recent_weight_a + recent_weight_b) / 2.0

        # Trust Score (dashboard)
        trust_ml = 1.0
        if experience_factor < 15:
            trust_ml *= 0.70
        elif experience_factor < 40:
            trust_ml *= 0.85

        if model_disagreement > 0.20:
            trust_ml *= 0.60
        elif model_disagreement > 0.12:
            trust_ml *= 0.80

        if elo_gap < 25:
            trust_ml *= 0.75

        trust_ml *= (0.85 + recency_factor * 0.15)

        ml_elo_same_direction = (
            (meta_prob > 0.5 and elo_prob_a > 0.5) or
            (meta_prob < 0.5 and elo_prob_a < 0.5)
        )
        if ml_elo_same_direction and model_disagreement < 0.10:
            trust_ml = min(trust_ml + 0.05, 0.95)

        trust_ml = float(np.clip(trust_ml, 0.20, 0.95))

        # (3) Shrinkage REAL hacia ancla calibrada (0.5 ↔ ELO)
        #     Antes uncertainty se calculaba y NO se aplicaba a prob_a.
        extreme = abs(meta_prob - 0.5)
        shrink_strength = 0.22 * (extreme ** 2)
        if experience_factor < 25:
            shrink_strength += 0.12 * (1.0 - (experience_factor / 25.0))
        if model_disagreement > 0.14:
            shrink_strength += 0.06
        if ml_elo_same_direction and model_disagreement < 0.08 and elo_gap > 50:
            shrink_strength *= 0.45
        uncertainty = float(np.clip(shrink_strength, 0.0, 0.22))

        # Ancla: mezcla 50/50 con ELO (más ELO si hay historial)
        elo_anchor_w = float(np.clip(0.35 + 0.45 * min(experience_factor / 40.0, 1.0), 0.35, 0.80))
        anchor = (1.0 - elo_anchor_w) * 0.5 + elo_anchor_w * elo_prob_a
        prob_a = (1.0 - uncertainty) * meta_prob + uncertainty * anchor

        prob_a = float(np.clip(prob_a, 0.02, 0.98))
        prob_b = 1.0 - prob_a

        confidence = self._confidence_score(
            prob_a,
            feat_a,
            feat_b
        )

        return MatchPrediction(

            player_a=feat_a.player_name,
            player_b=feat_b.player_name,

            prob_a=round(prob_a, 4),
            prob_b=round(prob_b, 4),

            expected_180s_a=round(feat_a.avg_180s, 2),
            expected_180s_b=round(feat_b.avg_180s, 2),

            expected_avg_a=round(feat_a.form_avg, 2),
            expected_avg_b=round(feat_b.form_avg, 2),

            model_version="v14_cal_blend",

            raw_prob=round(raw_prob_a, 4),

            elo_prob=round(elo_prob_a, 4),

            edge_vs_elo=round(
                prob_a - elo_prob_a,
                4
            ),

            confidence=round(confidence, 4),

            trust_ml=round(trust_ml, 4),

            model_disagreement=round(
                model_disagreement,
                4
            ),

            uncertainty=round(
                uncertainty,
                4
            ),
            edge_quality=(
                "LOW" if min(feat_a.n_matches, feat_b.n_matches) < 12 
                or model_disagreement > 0.18 
                or uncertainty > 0.12 
                or confidence < 0.22 
                else "HIGH" if min(feat_a.n_matches, feat_b.n_matches) >= 25 
                and model_disagreement <= 0.10 
                and confidence >= 0.35 
                else "MEDIUM"
            ),
            first_throw_edge_pp=round(ft_adj.edge_pp, 3),
            first_throw_closeness=round(ft_adj.closeness, 4),
            throw_neutral_prob=round(ft_adj.throw_neutral_prob, 4),
        )

    # ============================================================
    # LEGACY ALIAS
    # ============================================================

    def predict_match_v8(self, *args, **kwargs):

        return self.predict_match(
            *args,
            **kwargs
        )


if __name__ == "__main__":

    print("GRAVEL v12 ULTRA loaded.")

