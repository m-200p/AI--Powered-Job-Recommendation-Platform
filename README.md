# 🤖 AI-Powered Job Recommendation Platform

A full-stack AI job recommendation system using NLP, semantic embeddings, and a hybrid recommendation engine (content-based + collaborative filtering).

---

## 📁 Project Structure

```
ai-job-platform/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── recommender.py          # Hybrid recommendation engine
│   ├── preprocessor.py         # Text preprocessing & skill extraction
│   ├── embedder.py             # TF-IDF + semantic embeddings
│   ├── ranker.py               # Weighted final scoring & explainability
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.jsx
│       ├── index.jsx
│       ├── components/
│       │   ├── ResumeUploader.jsx
│       │   ├── JobCard.jsx
│       │   ├── ScoreBadge.jsx
│       │   └── ExplainPanel.jsx
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── Results.jsx
│       │   └── Dashboard.jsx
│       └── utils/
│           └── api.js
├── data/
│   ├── sample_resumes.json     # Sample resume data
│   └── sample_jobs.json        # Sample job listings
├── models/                     # Saved model artifacts (auto-generated)
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI
├── .gitignore
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.9+
- Node.js 18+
- pip & npm

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-job-platform.git
cd ai-job-platform
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```
Backend runs at: `http://localhost:5000`

### 3. Frontend Setup
```bash
cd frontend
npm install
npm start
```
Frontend runs at: `http://localhost:3000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/recommend` | Get job recommendations for a resume |
| `GET`  | `/api/jobs` | List all available jobs |
| `POST` | `/api/parse-resume` | Parse and extract skills from resume text |
| `GET`  | `/api/health` | Health check |

### Example Request
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python developer with 3 years experience in ML and NLP..."}'
```

---

## 🧠 How It Works

1. **Text Preprocessing** — Tokenization, lemmatization, stop-word removal, skill normalization
2. **Skill Extraction** — Named Entity Recognition (spaCy) extracts skills, degrees, experience
3. **Embedding** — TF-IDF + sentence-transformers produce hybrid feature vectors
4. **Similarity Scoring** — Cosine similarity between resume and job vectors
5. **Hybrid Ranking** — Weighted combination: `skill_match + experience + location + profile`
6. **Explainability** — Human-readable explanation for each recommendation

---

## 📊 Performance (from paper)

| Method | Precision | Recall | F1 | NDCG |
|--------|-----------|--------|----|------|
| Keyword TF-IDF | 0.71 | 0.68 | 0.69 | 0.66 |
| ML Classifier | 0.76 | 0.73 | 0.74 | 0.72 |
| Transformer Only | 0.82 | 0.79 | 0.80 | 0.78 |
| **Proposed Hybrid** | **0.88** | **0.85** | **0.86** | **0.84** |

---

## 🔮 Future Work
- Bias detection & fairness-aware ranking
- Multilingual resume support
- LLM-based career advice module
- Real-time job feed integration

---

## 📄 License
MIT
