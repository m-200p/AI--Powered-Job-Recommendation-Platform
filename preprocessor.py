"""
preprocessor.py
---------------
Text preprocessing and skill extraction pipeline.
Handles tokenization, lemmatization, stop-word removal,
named entity recognition, and skill normalization.
"""

import re
import string
import spacy
import nltk
from nltk.corpus import stopwords

# Download NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# ─── Skill Synonym Normalization Map ──────────────────────────────────────────
SKILL_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "dl": "deep learning",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "cv": "computer vision",
    "oop": "object oriented programming",
    "sql": "sql",
    "nosql": "nosql",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "nodejs",
    "vue.js": "vue",
    "rest api": "rest",
    "restful": "rest",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ci/cd": "continuous integration",
    "devops": "devops",
}

# ─── Known Technical Skills Vocabulary ────────────────────────────────────────
SKILL_KEYWORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    # Frameworks & Libraries
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "nodejs", "express", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "opencv",
    # Cloud & DevOps
    "aws", "azure", "google cloud platform", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "github actions",
    # Databases
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb",
    # AI/ML
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "transformers", "bert",
    "llm", "neural network", "data science", "mlops",
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum",
}


class Preprocessor:
    """Handles all text cleaning and skill extraction."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """Remove noise, normalize whitespace."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
        text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)      # Remove special chars
        text = re.sub(r"\s+", " ", text).strip()             # Collapse whitespace
        return text

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """Tokenize, remove stop words, and lemmatize."""
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 1
        ]
        return tokens

    def normalize_skills(self, skills: list[str]) -> list[str]:
        """Resolve synonyms to a canonical form."""
        normalized = []
        for skill in skills:
            skill_lower = skill.lower().strip()
            canonical = SKILL_SYNONYMS.get(skill_lower, skill_lower)
            normalized.append(canonical)
        return list(set(normalized))

    def extract_skills(self, text: str) -> list[str]:
        """Extract known skills from text via keyword matching + NER."""
        text_lower = text.lower()
        found_skills = []

        # Keyword matching against vocabulary
        for skill in SKILL_KEYWORDS:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found_skills.append(skill)

        # NER-based extraction (ORG entities often = tech companies/tools)
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                candidate = ent.text.lower().strip()
                if candidate in SKILL_KEYWORDS or candidate in SKILL_SYNONYMS:
                    found_skills.append(candidate)

        return self.normalize_skills(found_skills)

    def extract_experience_years(self, text: str) -> int:
        """Heuristically extract years of experience from text."""
        patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
            r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
            r"(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:work\s*)?experience",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0

    def extract_education(self, text: str) -> str:
        """Extract highest education level mentioned."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["phd", "ph.d", "doctorate"]):
            return "phd"
        if any(w in text_lower for w in ["master", "m.tech", "m.sc", "mba", "m.e"]):
            return "masters"
        if any(w in text_lower for w in ["bachelor", "b.tech", "b.sc", "b.e", "b.com"]):
            return "bachelors"
        return "other"

    def preprocess(self, text: str) -> dict:
        """
        Full preprocessing pipeline.
        Returns structured dict with cleaned text, tokens, skills, experience, education.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        skills = self.extract_skills(text)       # Use original for better matching
        experience = self.extract_experience_years(text)
        education = self.extract_education(text)

        return {
            "cleaned_text": cleaned,
            "tokens": tokens,
            "processed_text": " ".join(tokens),
            "skills": skills,
            "experience_years": experience,
            "education": education,
        }


# Singleton instance
preprocessor = Preprocessor()
