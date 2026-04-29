"""
ranker.py
---------
Weighted final scoring and explainability module.

FinalScore = w1*S_skills + w2*S_experience + w3*S_location + w4*S_semantic

Each recommendation includes a human-readable explanation (SHAP/LIME-inspired).
"""

from __future__ import annotations
import numpy as np


# ─── Default Weights (tunable) ────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "skill_match": 0.40,
    "semantic_similarity": 0.35,
    "experience_match": 0.15,
    "location_match": 0.10,
}


def skill_match_score(resume_skills: list[str], job_skills: list[str]) -> tuple[float, list[str]]:
    """
    Jaccard-based skill overlap score.
    Returns (score 0-1, list of matched skills).
    """
    if not job_skills:
        return 0.0, []
    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    matched = resume_set & job_set
    score = len(matched) / len(job_set)
    return round(score, 4), sorted(matched)


def experience_match_score(resume_years: int, job_min_years: int, job_max_years: int = 99) -> float:
    """
    Score how well resume experience fits the job's required range.
    Returns 1.0 for exact/over-qualified within range, degrades for under-qualification.
    """
    if resume_years >= job_min_years:
        return 1.0 if resume_years <= job_max_years else 0.85   # slightly penalize over-qual
    gap = job_min_years - resume_years
    return max(0.0, 1.0 - (gap * 0.2))


def location_match_score(resume_location: str, job_location: str) -> float:
    """
    Simple string-based location match (can be replaced with geocoding).
    """
    if not resume_location or not job_location:
        return 0.5   # neutral when unknown
    r = resume_location.lower().strip()
    j = job_location.lower().strip()
    if r == j:
        return 1.0
    if "remote" in j or "remote" in r:
        return 0.9
    # Check city/country overlap
    r_parts = set(r.replace(",", " ").split())
    j_parts = set(j.replace(",", " ").split())
    overlap = r_parts & j_parts
    return min(1.0, len(overlap) / max(len(j_parts), 1))


def compute_final_score(
    semantic_sim: float,
    skill_score: float,
    experience_score: float,
    location_score: float,
    weights: dict = None,
) -> float:
    """Weighted combination of all sub-scores."""
    w = weights or DEFAULT_WEIGHTS
    score = (
        w["skill_match"] * skill_score
        + w["semantic_similarity"] * semantic_sim
        + w["experience_match"] * experience_score
        + w["location_match"] * location_score
    )
    return round(score, 4)


def generate_explanation(
    job: dict,
    matched_skills: list[str],
    missing_skills: list[str],
    semantic_sim: float,
    experience_score: float,
    location_score: float,
    final_score: float,
) -> dict:
    """
    Generate a human-readable explanation for why a job was recommended.
    Inspired by SHAP/LIME-style feature attribution.
    """
    reasons = []
    warnings = []

    # Skill match explanation
    if matched_skills:
        top_skills = matched_skills[:4]
        reasons.append(f"Strong skill match: {', '.join(top_skills)}")
    if missing_skills:
        top_missing = missing_skills[:3]
        warnings.append(f"Skills to develop: {', '.join(top_missing)}")

    # Semantic similarity
    if semantic_sim >= 0.75:
        reasons.append("Your profile closely aligns with this role's description")
    elif semantic_sim >= 0.50:
        reasons.append("Moderate alignment with the role's requirements")

    # Experience
    if experience_score >= 0.9:
        reasons.append("Your experience level matches the requirement")
    elif experience_score >= 0.6:
        warnings.append("You may be slightly under the experience requirement")

    # Location
    if location_score >= 0.9:
        reasons.append("Location is a strong match or remote-friendly")
    elif location_score < 0.4:
        warnings.append("Location may require relocation or remote arrangement")

    return {
        "final_score": final_score,
        "score_display": f"{final_score:.0%}",
        "reasons": reasons,
        "warnings": warnings,
        "score_breakdown": {
            "skill_match": f"{matched_skills.__len__()} of {(matched_skills.__len__() + missing_skills.__len__())} required skills matched",
            "semantic_alignment": f"{semantic_sim:.0%}",
            "experience_fit": f"{experience_score:.0%}",
            "location_fit": f"{location_score:.0%}",
        },
    }


def rank_jobs(
    resume_profile: dict,
    resume_vector: np.ndarray,
    jobs: list[dict],
    job_vectors: list[np.ndarray],
    weights: dict = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Rank all jobs for a resume using the hybrid scoring formula.

    resume_profile: output of Preprocessor.preprocess() + location field
    resume_vector: hybrid embedding of the resume
    jobs: list of job dicts (from data/sample_jobs.json)
    job_vectors: precomputed embeddings for each job

    Returns top_k ranked jobs with scores and explanations.
    """
    results = []

    for job, job_vec in zip(jobs, job_vectors):
        # 1. Semantic similarity
        semantic_sim = float(np.dot(resume_vector, job_vec))
        semantic_sim = max(0.0, min(1.0, semantic_sim))

        # 2. Skill match
        job_required_skills = job.get("required_skills", [])
        skill_score, matched = skill_match_score(
            resume_profile.get("skills", []), job_required_skills
        )
        missing = [s for s in job_required_skills if s.lower() not in
                   {m.lower() for m in matched}]

        # 3. Experience match
        exp_score = experience_match_score(
            resume_profile.get("experience_years", 0),
            job.get("min_experience", 0),
            job.get("max_experience", 99),
        )

        # 4. Location match
        loc_score = location_match_score(
            resume_profile.get("location", ""),
            job.get("location", ""),
        )

        # 5. Final score
        final = compute_final_score(semantic_sim, skill_score, exp_score, loc_score, weights)

        # 6. Explanation
        explanation = generate_explanation(
            job, matched, missing, semantic_sim, exp_score, loc_score, final
        )

        results.append({
            **job,
            "explanation": explanation,
            "match_score": final,
        })

    # Sort descending by match_score
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]
