"""
recommender.py
--------------
Core hybrid recommendation engine.
Orchestrates: preprocessing → embedding → ranking.

Supports:
  - Cold-start (no user history): pure semantic matching
  - Warm users: semantic + collaborative weighting boost
"""

from __future__ import annotations
import json
import os
import numpy as np

from preprocessor import preprocessor
from embedder import embedder
from ranker import rank_jobs

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JOBS_FILE = os.path.join(DATA_DIR, "sample_jobs.json")


class JobRecommender:
    """
    Main recommender class.
    On init, loads job data, fits embedder, and precomputes job vectors.
    """

    def __init__(self):
        self.jobs: list[dict] = []
        self.job_vectors: list[np.ndarray] = []
        self._user_history: dict[str, list[str]] = {}   # user_id -> list of applied job_ids
        self._loaded = False

    def load_and_index(self):
        """Load jobs from disk, fit embedder, precompute job vectors."""
        with open(JOBS_FILE, "r") as f:
            self.jobs = json.load(f)

        # Preprocess each job
        job_texts = []
        for job in self.jobs:
            proc = preprocessor.preprocess(
                job.get("title", "") + " " + job.get("description", "") + " " +
                " ".join(job.get("required_skills", []))
            )
            job["_processed"] = proc
            job_texts.append(proc["processed_text"])

        # Fit TF-IDF on all job texts
        embedder.fit(job_texts)

        # Precompute job vectors
        self.job_vectors = [embedder.embed_single(t) for t in job_texts]
        self._loaded = True
        print(f"[Recommender] Indexed {len(self.jobs)} jobs.")

    def recommend(
        self,
        resume_text: str,
        location: str = "",
        user_id: str = None,
        top_k: int = 10,
    ) -> dict:
        """
        Main recommendation method.

        resume_text: raw resume text (pasted or extracted from PDF)
        location: candidate's preferred location
        user_id: optional, for collaborative boosting
        top_k: number of results to return

        Returns dict with profile, ranked recommendations, and metadata.
        """
        if not self._loaded:
            self.load_and_index()

        # 1. Parse resume
        resume_profile = preprocessor.preprocess(resume_text)
        resume_profile["location"] = location

        # 2. Embed resume
        resume_vector = embedder.embed_single(resume_profile["processed_text"])

        # 3. Collaborative boost (warm-start): boost previously-interacted jobs' domains
        job_vectors = self.job_vectors
        if user_id and user_id in self._user_history:
            job_vectors = self._apply_collaborative_boost(resume_vector, user_id)

        # 4. Rank
        ranked = rank_jobs(
            resume_profile=resume_profile,
            resume_vector=resume_vector,
            jobs=self.jobs,
            job_vectors=job_vectors,
            top_k=top_k,
        )

        # Clean internal fields before returning
        for job in ranked:
            job.pop("_processed", None)

        return {
            "resume_profile": {
                "skills": resume_profile["skills"],
                "experience_years": resume_profile["experience_years"],
                "education": resume_profile["education"],
                "location": location,
            },
            "recommendations": ranked,
            "total_jobs_indexed": len(self.jobs),
            "cold_start": user_id is None or user_id not in self._user_history,
        }

    def _apply_collaborative_boost(
        self, resume_vector: np.ndarray, user_id: str
    ) -> list[np.ndarray]:
        """
        Collaborative signal: slightly boost similarity for jobs in domains
        the user has previously interacted with.
        Boost factor = 1.05 for matching domain jobs.
        """
        history = self._user_history.get(user_id, [])
        interacted_domains = set()
        for job in self.jobs:
            if job.get("id") in history:
                interacted_domains.add(job.get("domain", ""))

        boosted = []
        for job, vec in zip(self.jobs, self.job_vectors):
            if job.get("domain", "") in interacted_domains:
                boosted.append(vec * 1.05)   # small collaborative boost
            else:
                boosted.append(vec)
        return boosted

    def record_interaction(self, user_id: str, job_id: str):
        """Record a user click/application for collaborative filtering."""
        if user_id not in self._user_history:
            self._user_history[user_id] = []
        if job_id not in self._user_history[user_id]:
            self._user_history[user_id].append(job_id)

    def parse_resume_only(self, resume_text: str) -> dict:
        """Return parsed resume profile without recommendations."""
        profile = preprocessor.preprocess(resume_text)
        return {
            "skills": profile["skills"],
            "experience_years": profile["experience_years"],
            "education": profile["education"],
        }

    def get_all_jobs(self) -> list[dict]:
        if not self._loaded:
            self.load_and_index()
        return [{k: v for k, v in j.items() if k != "_processed"} for j in self.jobs]


# Singleton
recommender = JobRecommender()
