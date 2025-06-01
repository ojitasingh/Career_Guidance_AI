from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# === Configuration === #
class Config:
    NLP_MODEL_NAME = "all-MiniLM-L6-v2"
    MODEL_VERSION = "1.0.0"
    MAX_RECOMMENDATIONS = 5
    MIN_SIMILARITY_SCORE = 0.3
    CACHE_SIZE = 1024

# === Data Models === #
class Career:
    def __init__(self, data: Dict):
        self.name = data["career"]
        self.streams = data["streams"]
        self.skills = data["skills"]
        self.description = data["description"]
        self.environments = data.get("environments", [])
        self.motivations = data.get("motivations", [])
        self.exams = data.get("exams", [])
        self.growth = data.get("growth", "medium")  # low, medium, high
        self.salary_range = data.get("salary_range", (3, 8))  # in LPA
        self.demand = data.get("demand", "high")  # low, medium, high
        
    def to_dict(self) -> Dict:
        return {
            "career": self.name,
            "streams": self.streams,
            "skills": self.skills,
            "description": self.description,
            "environments": self.environments,
            "motivations": self.motivations,
            "exams": self.exams,
            "growth": self.growth,
            "salary_range": self.salary_range,
            "demand": self.demand
        }

# === Model Management === #
class ModelManager:
    def __init__(self):
        self.clf = None
        self.nlp_model = None
        self.careers = []
        self.cluster_embeddings = {}
        self.load_models()
        
    def load_models(self):
        """Load or initialize all required models"""
        try:
            # Load or train classifier
            clf_path = MODEL_DIR / "career_classifier.pkl"
            if clf_path.exists():
                with open(clf_path, "rb") as f:
                    self.clf = pickle.load(f)
                logger.info("Loaded pre-trained classifier")
            else:
                self.train_classifier()
                
            # Load NLP model
            self.nlp_model = SentenceTransformer(Config.NLP_MODEL_NAME)
            logger.info(f"Loaded NLP model: {Config.NLP_MODEL_NAME}")
            
            # Load career data
            self.load_career_data()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def train_classifier(self):
        """Train or retrain the stream classifier"""
        try:
            # In a real app, you would load actual training data
            stream_map = {'science': 0, 'commerce': 1, 'arts': 2, 'engineering': 3}
            
            # Mock training data (replace with real data)
            X_train = [
                [0, 0, 0, 0, 0],  # science
                [1, 1, 1, 3, 0],  # commerce
                [2, 2, 2, 1, 1],  # arts
                [3, 0, 3, 2, 2]   # engineering
            ]
            y_train = [0, 1, 2, 3]
            
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.clf.fit(X_train, y_train)
            
            # Save the trained model
            with open(MODEL_DIR / "career_classifier.pkl", "wb") as f:
                pickle.dump(self.clf, f)
                
            logger.info("Trained and saved new classifier model")
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise
            
    def load_career_data(self):
        """Load career data from file or initialize default"""
        try:
            data_path = DATA_DIR / "careers.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    career_list = json.load(f)
            else:
                # Default career data if file doesn't exist
                career_list = [
                    {
                        "career": "Software Engineer",
                        "streams": ["science", "engineering"],
                        "skills": ["programming", "problem solving", "algorithms"],
                        "description": "Build software applications and systems.",
                        "environments": ["dynamic", "quiet"],
                        "motivations": ["income", "innovation"],
                        "exams": ["JEE", "GATE"],
                        "growth": "high",
                        "salary_range": [6, 40],
                        "demand": "high"
                    },
                    # ... (other careers as shown in previous example)
                ]
                with open(data_path, "w") as f:
                    json.dump(career_list, f, indent=2)
                    
            self.careers = [Career(data) for data in career_list]
            
            # Precompute cluster embeddings
            career_clusters = {
                "engineering": ["software", "technology", "engineer", "developer"],
                "commerce": ["finance", "accounting", "business", "tax"],
                "arts": ["design", "music", "drawing", "literature"],
                "government": ["ias", "ips", "public", "civil services"],
                "medical": ["doctor", "medicine", "health", "patient"],
                "education": ["teacher", "professor", "education", "learning"]
            }
            
            self.cluster_embeddings = {
                key: self.nlp_model.encode(" ".join(words)) 
                for key, words in career_clusters.items()
            }
            
            logger.info(f"Loaded career data with {len(self.careers)} careers")
            
        except Exception as e:
            logger.error(f"Error loading career data: {str(e)}")
            raise

# Initialize model manager
model_manager = ModelManager()

# === Utility Functions === #
@lru_cache(maxsize=Config.CACHE_SIZE)
def encode_answers(answers: Dict) -> List[int]:
    """Encode user answers to numerical features"""
    answer_map = {
        'q1': {"logical": 0, "numerical": 1, "creative": 2, "verbal": 3},
        'q2': {"visual": 0, "auditory": 1, "kinesthetic": 2, "reading": 3},
        'q3': {"science": 0, "commerce": 1, "arts": 2, "engineering": 3},
        'q4': {"structured": 0, "creative": 1, "quiet": 2, "dynamic": 3},
        'q5': {"income": 0, "helping": 1, "innovation": 2, "stability": 3}
    }
    
    return [
        answer_map['q1'].get(answers.get('q1', ''), 0),
        answer_map['q2'].get(answers.get('q2', ''), 0),
        answer_map['q3'].get(answers.get('q3', ''), 0),
        answer_map['q4'].get(answers.get('q4', ''), 0),
        answer_map['q5'].get(answers.get('q5', ''), 0)
    ]

def predict_stream(encoded_answers: List[int]) -> str:
    """Predict dominant stream from encoded answers"""
    try:
        prediction = model_manager.clf.predict([encoded_answers])[0]
        stream_map = {0: "science", 1: "commerce", 2: "arts", 3: "engineering"}
        return stream_map.get(prediction, "science")
    except Exception as e:
        logger.error(f"Error predicting stream: {str(e)}")
        return "science"  # fallback

@lru_cache(maxsize=Config.CACHE_SIZE)
def detect_clusters(aspiration_text: str) -> List[str]:
    """Detect career clusters from aspirations text"""
    try:
        if not aspiration_text.strip():
            return []
            
        vec = model_manager.nlp_model.encode(aspiration_text)
        scores = {
            k: float(util.cos_sim(vec, v)[0]) 
            for k, v in model_manager.cluster_embeddings.items()
        }
        
        # Filter and sort clusters by score
        return [
            k for k, v in sorted(scores.items(), key=lambda x: -x[1])
            if v >= Config.MIN_SIMILARITY_SCORE
        ][:2]
        
    except Exception as e:
        logger.error(f"Error detecting clusters: {str(e)}")
        return []

def calculate_match_score(
    career: Career,
    skills: set,
    preferred_env: Optional[str] = None,
    preferred_motivation: Optional[str] = None
) -> Tuple[int, List[str]]:
    """Calculate match score between user and career"""
    required_skills = set(s.lower() for s in career.skills)
    skill_match = len(required_skills & skills)
    
    # Bonus points for environment and motivation match
    extra_points = 0
    if preferred_env and preferred_env in career.environments:
        extra_points += 1
    if preferred_motivation and preferred_motivation in career.motivations:
        extra_points += 1
        
    missing_skills = list(required_skills - skills)
    return (skill_match + extra_points, missing_skills)

def recommend_careers(
    stream: str,
    clusters: List[str],
    skills: List[str],
    preferred_env: Optional[str] = None,
    preferred_motivation: Optional[str] = None
) -> List[Dict]:
    """Recommend careers based on multiple factors"""
    try:
        skills_set = set(s.strip().lower() for s in skills)
        results = []
        
        for career in model_manager.careers:
            # Check stream match (either dominant stream or detected clusters)
            stream_match = (stream in career.streams or 
                          any(c in career.streams for c in clusters))
            
            if stream_match:
                match_score, missing_skills = calculate_match_score(
                    career, skills_set, preferred_env, preferred_motivation
                )
                
                if match_score > 0:  # Only include careers with some match
                    results.append({
                        "career": career.name,
                        "description": career.description,
                        "match_score": match_score,
                        "missing_skills": missing_skills,
                        "streams": career.streams,
                        "exams": career.exams,
                        "growth": career.growth,
                        "salary_range": career.salary_range,
                        "demand": career.demand
                    })
        
        # Sort by match score (descending) and other factors
        results.sort(
            key=lambda x: (
                -x["match_score"],
                -x["salary_range"][1],  # max salary potential
                -1 if x["demand"] == "high" else (-0.5 if x["demand"] == "medium" else 0),
                -1 if x["growth"] == "high" else (-0.5 if x["growth"] == "medium" else 0)
            )
        )
        
        return results[:Config.MAX_RECOMMENDATIONS]
        
    except Exception as e:
        logger.error(f"Error recommending careers: {str(e)}")
        return []

# === API Endpoints === #
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    start_time = datetime.now()
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        answers = data.get("answers", {})
        aspirations = data.get("aspirations", "")
        skills = data.get("skills", [])
        
        # Validate inputs
        if not isinstance(answers, dict) or len(answers) < 5:
            return jsonify({"error": "Invalid or incomplete answers"}), 400
            
        if not aspirations.strip():
            return jsonify({"error": "Aspirations text is required"}), 400
            
        if not skills:
            return jsonify({"error": "At least one skill is required"}), 400
            
        # Process the request
        encoded = encode_answers(answers)
        predicted_stream = predict_stream(encoded)
        clusters = detect_clusters(aspirations.lower())
        
        recommendations = recommend_careers(
            predicted_stream,
            clusters,
            skills,
            answers.get("q4"),
            answers.get("q5")
        )
        
        response = {
            "meta": {
                "version": Config.MODEL_VERSION,
                "processing_time": (datetime.now() - start_time).total_seconds()
            },
            "results": {
                "predicted_stream": predicted_stream,
                "career_clusters": clusters,
                "recommendations": recommendations
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@app.route('/api/careers', methods=['GET'])
def list_careers():
    """Endpoint to list all available careers (for frontend reference)"""
    try:
        return jsonify({
            "careers": [career.to_dict() for career in model_manager.careers]
        })
    except Exception as e:
        logger.error(f"Error listing careers: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": Config.MODEL_VERSION,
        "model_loaded": model_manager.clf is not None,
        "careers_loaded": len(model_manager.careers) > 0
    })

# === Main === #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)