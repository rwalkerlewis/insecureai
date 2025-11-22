"""
Module 9: Vector-Based Classification using Ollama Embeddings
Learn: Embedding generation, cosine similarity, semantic classification
"""

import numpy as np
import requests
import json
from typing import List, Dict

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"  # Fast and efficient embedding model

#categories = [
#    "mathematical calculation or equation",
#    "time or date related question",
#    "location or place query",
#    "person or biographical question",
#    "definition or explanation",
#    "how-to or instructional request",
#    "comparison or evaluation",
#    #"general knowledge question",
#]

#categories = [
#    "mathematical",
#    "time",
#    "location",
#    "person",
#    "definition",
#    "how-to",
#    "comparison",
#    "place",
#]

categories = {
    "mathematics : solving mathematical equations and performing numerical calculations with arithmetic operations",
    
    "time : asking about dates, times, durations, and when events occurred or will occur",
    
    "place : asking about geographical locations, addresses, and where things are physically located",
    
    "person : asking about people's identities, biographies, and who someone is or was",
    
    "definition : asking for explanations, meanings, and definitions of concepts and terms",
    
    "how-to : asking for instructions, methods, and step-by-step guidance on doing tasks",
    
    "comparison : asking about differences, similarities, and comparing multiple options or choices",
    
    "yes-no : questions that can be answered with yes or no, true or false responses",
    
    "opinion : asking for subjective views, recommendations, preferences, and personal judgments",
    
    "factual : asking for objective facts, data, statistics, and verifiable information"
}



def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate embedding for text using Ollama API
    
    Args:
        text: Input text to embed
        model: Ollama embedding model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to get embedding from Ollama: {e}")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1, higher = more similar)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance (L2 norm) between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (lower = more similar)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return np.linalg.norm(vec1_np - vec2_np)


def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Manhattan distance (L1 norm) between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Manhattan distance (lower = more similar)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return np.sum(np.abs(vec1_np - vec2_np))


class VectorClassifier:
    """
    Classify prompts using vector embeddings and multiple similarity metrics
    """
    
    def __init__(self, categories: List[str], model: str = EMBEDDING_MODEL, metric: str = "cosine"):
        """
        Initialize classifier with categories
        
        Args:
            categories: List of category labels
            model: Ollama embedding model to use
            metric: Similarity metric to use ('cosine', 'euclidean', 'manhattan')
        """
        self.categories = categories
        self.model = model
        self.metric = metric
        self.category_embeddings = None
        
        # Map metric names to functions and whether higher is better
        self.metric_functions = {
            "cosine": (cosine_similarity, True),      # Higher is better
            "euclidean": (euclidean_distance, False), # Lower is better
            "manhattan": (manhattan_distance, False)  # Lower is better
        }
        
        if metric not in self.metric_functions:
            raise ValueError(f"Unknown metric: {metric}. Choose from: {list(self.metric_functions.keys())}")
        
        print(f"🔄 Initializing VectorClassifier with {len(categories)} categories...")
        print(f"   Metric: {metric}")
        self._precompute_category_embeddings()
        print(f"✅ Category embeddings cached\n")
    
    def _precompute_category_embeddings(self):
        """Pre-compute embeddings for all categories to speed up classification"""
        self.category_embeddings = []
        for category in self.categories:
            embedding = get_embedding(category, self.model)
            self.category_embeddings.append(embedding)
    
    def classify(self, prompt: str) -> Dict:
        """
        Classify a prompt by finding the most similar category
        
        Args:
            prompt: Input text to classify
            
        Returns:
            Dictionary with classification results including category, confidence, and all scores
        """
        # Generate embedding for the prompt
        prompt_embedding = get_embedding(prompt, self.model)
        
        # Get the metric function and whether higher is better
        metric_func, higher_is_better = self.metric_functions[self.metric]
        
        # Calculate similarity scores for each category
        similarities = []
        for category, category_embedding in zip(self.categories, self.category_embeddings):
            score = metric_func(prompt_embedding, category_embedding)
            similarities.append({
                'category': category,
                'score': score
            })
        
        # Sort by score (best first)
        similarities.sort(key=lambda x: x['score'], reverse=higher_is_better)
        
        return {
            'category': similarities[0]['category'],
            'confidence': similarities[0]['score'],
            'all_scores': {item['category']: item['score'] for item in similarities}
        }


def main():
    """Demo the vector-based classifier with multiple metrics"""
    print("=== Module 9: Vector-Based Classification ===\n")
    
    # Test prompts
    test_prompts = [
        "What's 15% of 200?",
        "What's 4 * 12?",
        "What is 2 + 2?",
        "1 + 1",
        "Who is the current President of the United States?",
        "How do I bake a chocolate cake?",
    ]
    
    # Compare all three metrics
    metrics = ["cosine", "euclidean", "manhattan"]
    
    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"  📊 Using {metric.upper()} metric")
        print(f"{'='*70}\n")
        
        classifier = VectorClassifier(categories, metric=metric)
        
        for prompt in test_prompts:
            result = classifier.classify(prompt)
            print(f"Prompt: \"{prompt}\"")
            print(f"  ➜ Category: {result['category']}")
            print(f"  ➜ Score: {result['confidence']:.4f}")
            print(f"  ➜ Top 3 matches:")
            for i, (category, score) in enumerate(list(result['all_scores'].items())[:3], 1):
                print(f"     {i}. {category}: {score:.4f}")
            print()
    
    print("\n" + "="*70)
    print("  ✓ Module 9 complete - Metric comparison done")
    print("="*70)


if __name__ == "__main__":
    main()