"""Judge-human agreement statistics. UNMEASURED for this submission -- see
judge_prompt.py docstring. Implemented and unit-tested so it is a single command away
once judge scores + human scores on the same rubric both exist.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score


def exact_agreement(judge_scores: list[int], human_scores: list[int]) -> float:
    return float(np.mean([j == h for j, h in zip(judge_scores, human_scores)]))


def within_one_agreement(judge_scores: list[int], human_scores: list[int]) -> float:
    return float(np.mean([abs(j - h) <= 1 for j, h in zip(judge_scores, human_scores)]))


def weighted_kappa(judge_scores: list[int], human_scores: list[int]) -> float:
    return float(cohen_kappa_score(judge_scores, human_scores, weights="linear"))


def agreement_report(judge_scores: list[int], human_scores: list[int]) -> dict:
    assert len(judge_scores) == len(human_scores) and len(judge_scores) > 0, \
        "judge_scores and human_scores must be equal-length and non-empty"
    return {
        "n": len(judge_scores),
        "exact_agreement": round(exact_agreement(judge_scores, human_scores), 4),
        "within_one_agreement": round(within_one_agreement(judge_scores, human_scores), 4),
        "weighted_kappa": round(weighted_kappa(judge_scores, human_scores), 4),
    }
