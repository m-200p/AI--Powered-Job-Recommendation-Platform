"""
app.py
------
Flask REST API for the AI-Powered Job Recommendation Platform.
Run with: python app.py
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommender

app = Flask(__name__)
CORS(app)   # Allow frontend at localhost:3000

# Pre-load and index jobs on startup
recommender.load_and_index()


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "jobs_indexed": len(recommender.jobs)})


# ─── Get All Jobs ─────────────────────────────────────────────────────────────
@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    """Return all available job listings."""
    jobs = recommender.get_all_jobs()
    return jsonify({"jobs": jobs, "total": len(jobs)})


# ─── Parse Resume ─────────────────────────────────────────────────────────────
@app.route("/api/parse-resume", methods=["POST"])
def parse_resume():
    """
    Extract structured info from raw resume text.
    Body: { "resume_text": "..." }
    """
    data = request.get_json()
    if not data or "resume_text" not in data:
        return jsonify({"error": "resume_text is required"}), 400

    resume_text = data["resume_text"].strip()
    if not resume_text:
        return jsonify({"error": "resume_text cannot be empty"}), 400

    try:
        profile = recommender.parse_resume_only(resume_text)
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Recommend Jobs ───────────────────────────────────────────────────────────
@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Get personalized job recommendations.

    Body:
    {
        "resume_text": "...",       # required
        "location": "Bangalore",   # optional
        "user_id": "user123",      # optional (for collaborative filtering)
        "top_k": 10                # optional (default 10)
    }
    """
    data = request.get_json()
    if not data or "resume_text" not in data:
        return jsonify({"error": "resume_text is required"}), 400

    resume_text = data["resume_text"].strip()
    if not resume_text:
        return jsonify({"error": "resume_text cannot be empty"}), 400

    location = data.get("location", "")
    user_id = data.get("user_id", None)
    top_k = min(int(data.get("top_k", 10)), 50)

    try:
        result = recommender.recommend(
            resume_text=resume_text,
            location=location,
            user_id=user_id,
            top_k=top_k,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Record Interaction (for collaborative filtering) ─────────────────────────
@app.route("/api/interact", methods=["POST"])
def interact():
    """
    Record a user applying to / clicking a job.
    Body: { "user_id": "...", "job_id": "..." }
    """
    data = request.get_json()
    user_id = data.get("user_id")
    job_id = data.get("job_id")
    if not user_id or not job_id:
        return jsonify({"error": "user_id and job_id required"}), 400
    recommender.record_interaction(user_id, job_id)
    return jsonify({"status": "recorded"})


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
