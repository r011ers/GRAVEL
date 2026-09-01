"""
╔══════════════════════════════════════════════════════════════╗
║  GRAVEL — first_throw.py                                     ║
║  Calibración empírica del saque (LOCAL) en MSS Best of 7     ║
║  Fuente: outputs/first_throw_backtest (n=19,371)             ║
╚══════════════════════════════════════════════════════════════╝

Hallazgos clave del backtest:
  - WR global local (1er saque): 52.34%  → edge +2.34 pp
  - Emparejados |Δavg|≤2:         55.06%  → edge +5.06 pp
  - Casi idénticos |Δavg|<1:      55.64%  → edge +5.64 pp
  - Deciders (7 legs):            59.97%  (saque en leg impar final)

En predicción pre-partido no conocemos Δavg observado; usamos
cercanía de ELO / form avg / winrate para interpolar el boost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── Constantes empíricas (backtest MSS Bo7, 2026-08-09) ─────
N_MATCHES = 19371
WR_GLOBAL = 0.5233596613494399
WR_CLOSE = 0.5505787619526925   # |Δavg| ≤ 2
WR_PURE = 0.5564                # |Δavg| < 1

# Fiabilidad del prior (evita sobreajuste al backtest)
RELIABILITY = 0.92


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Log-odds del first thrower vs 50/50
LOGODDS_GLOBAL = _logit(WR_GLOBAL)          # ≈ +0.0935
LOGODDS_CLOSE = _logit(WR_CLOSE)            # ≈ +0.203
LOGODDS_PURE = _logit(WR_PURE)              # ≈ +0.227
# Residual mínimo cuando el mismatch de nivel es enorme
LOGODDS_FLOOR = 0.025


@dataclass(frozen=True)
class FirstThrowAdjustment:
    """Resultado de la calibración de saque."""
    prob: float
    closeness: float
    logodds_delta: float
    edge_pp: float          # signed, perspectiva player_a
    throw_neutral_prob: float


def match_closeness(
    elo_a: float,
    elo_b: float,
    avg_a: float,
    avg_b: float,
    wr_a: float = 0.5,
    wr_b: float = 0.5,
) -> float:
    """
    Cercanía de nivel pre-partido ∈ [0, 1].
    1.0 = empatados (máximo peso del saque).
    0.0 = mismatch grande (saque casi residual).
    """
    elo_gap = abs(float(elo_a) - float(elo_b))
    avg_gap = abs(float(avg_a) - float(avg_b))
    wr_gap = abs(float(wr_a) - float(wr_b))

    # Kernels gaussianos calibrados a la escala típica MSS
    c_elo = math.exp(-((elo_gap / 90.0) ** 2))   # ~0.5 a ±75 ELO
    c_avg = math.exp(-((avg_gap / 3.5) ** 2))    # ~0.5 a ±3 pts de avg
    c_wr = math.exp(-((wr_gap / 0.12) ** 2))     # ~0.5 a ±12 pp WR

    closeness = 0.50 * c_elo + 0.35 * c_avg + 0.15 * c_wr
    return float(min(max(closeness, 0.0), 1.0))


def first_throw_logodds_delta(first_throw_a: bool, closeness: float) -> float:
    """
    Boost en log-odds para player_a.
    Positivo si A saca primero; negativo si saca B.
    Interpola FLOOR → CLOSE según cercanía de nivel.
    """
    c = min(max(float(closeness), 0.0), 1.0)
    # Curva suave: más peso al régimen "close" cuando c alto
    delta_mag = LOGODDS_FLOOR + (LOGODDS_CLOSE - LOGODDS_FLOOR) * (c ** 0.85)
    delta_mag *= RELIABILITY
    return delta_mag if first_throw_a else -delta_mag


def apply_logodds_shift(prob: float, delta: float) -> float:
    """p' = σ(logit(p) + delta), clip a (0.02, 0.98)."""
    p = min(max(float(prob), 0.02), 0.98)
    return float(min(max(_sigmoid(_logit(p) + delta), 0.02), 0.98))


def throw_neutral_prob(prob_ft_a: float, prob_ft_b: float) -> float:
    """
    Probabilidad de A eliminando el sesgo de saque del modelo.
    media de P(A|A saca) y P(A|B saca).
    """
    return float(min(max(0.5 * (float(prob_ft_a) + float(prob_ft_b)), 0.02), 0.98))


def calibrate_first_throw(
    prob_if_a_throws: float,
    prob_if_b_throws: float,
    first_throw_a: bool,
    elo_a: float,
    elo_b: float,
    avg_a: float,
    avg_b: float,
    wr_a: float = 0.5,
    wr_b: float = 0.5,
) -> FirstThrowAdjustment:
    """
    Sustituye el componente de saque aprendido por el ML
    por el prior empírico del backtest MSS Bo7.

    1) p_neutral = media simétrica FT
    2) δ = log-odds empírico según cercanía de nivel
    3) p_final = σ(logit(p_neutral) + δ)
    """
    p_n = throw_neutral_prob(prob_if_a_throws, prob_if_b_throws)
    closeness = match_closeness(elo_a, elo_b, avg_a, avg_b, wr_a, wr_b)
    delta = first_throw_logodds_delta(first_throw_a, closeness)
    p_final = apply_logodds_shift(p_n, delta)

    # edge en probabilidad (pp) respecto al neutral
    edge_pp = (p_final - p_n) * 100.0

    return FirstThrowAdjustment(
        prob=p_final,
        closeness=closeness,
        logodds_delta=delta,
        edge_pp=edge_pp,
        throw_neutral_prob=p_n,
    )


def calibrate_elo_with_throw(
    elo_prob_a: float,
    first_throw_a: bool,
    closeness: float,
) -> float:
    """Aplica el mismo prior de saque a la rama ELO (throw-neutral por construcción)."""
    delta = first_throw_logodds_delta(first_throw_a, closeness)
    return apply_logodds_shift(elo_prob_a, delta)
