"""
Dataset module for loading and managing text data.
"""
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def load_texts(data_path: str = "data/texts.csv") -> Tuple[List[str], pd.DataFrame]:
    """
    Load texts from CSV file.
    
    Args:
        data_path: Path to the texts CSV file
        
    Returns:
        texts: List of text strings
        df: DataFrame containing all text data
    """
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    return texts, df


def get_texts_by_concept(df: pd.DataFrame, concept: str) -> List[str]:
    """
    Get all texts for a specific concept.
    
    Args:
        df: DataFrame containing text data
        concept: Concept name
        
    Returns:
        List of texts for the concept
    """
    return df[df['concept'] == concept]['text'].tolist()


def get_texts_by_template(df: pd.DataFrame, template_id: int) -> List[str]:
    """
    Get all texts for a specific template.
    
    Args:
        df: DataFrame containing text data
        template_id: Template ID (0-5)
        
    Returns:
        List of texts for the template
    """
    return df[df['template_id'] == template_id]['text'].tolist()


def generate_sample_data(output_path: str = "data/texts.csv", num_concepts: int = 150) -> None:
    """
    Generate sample dataset with concepts and templates.
    
    Args:
        output_path: Path to save the CSV file
        num_concepts: Number of concepts to generate
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample concepts (in a real scenario, these would be domain-specific)
    concepts = [
        # Abstract concepts
        "love", "happiness", "freedom", "justice", "truth", "beauty", "wisdom", "courage",
        "knowledge", "peace", "hope", "faith", "trust", "loyalty", "honesty", "integrity",
        # Emotions
        "joy", "sadness", "anger", "fear", "surprise", "disgust", "anticipation", "acceptance",
        "excitement", "anxiety", "calm", "stress", "relief", "pride", "shame", "guilt",
        # Natural objects
        "mountain", "river", "ocean", "forest", "desert", "sky", "sun", "moon",
        "star", "cloud", "rain", "snow", "wind", "fire", "earth", "water",
        # Animals
        "dog", "cat", "bird", "fish", "elephant", "lion", "tiger", "bear",
        "horse", "cow", "sheep", "pig", "chicken", "duck", "rabbit", "mouse",
        # Plants
        "tree", "flower", "grass", "rose", "lily", "orchid", "cactus", "fern",
        "oak", "pine", "maple", "willow", "bamboo", "moss", "ivy", "vine",
        # Foods
        "apple", "banana", "orange", "grape", "strawberry", "watermelon", "bread", "rice",
        "meat", "fish", "vegetable", "fruit", "cheese", "milk", "egg", "sugar",
        # Colors
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "gold", "silver", "bronze", "violet", "indigo",
        # Shapes
        "circle", "square", "triangle", "rectangle", "pentagon", "hexagon", "sphere", "cube",
        "pyramid", "cylinder", "cone", "prism", "oval", "diamond", "star shape", "heart shape",
        # Objects
        "table", "chair", "door", "window", "book", "pen", "paper", "computer",
        "phone", "car", "bicycle", "house", "building", "bridge", "road", "path",
        # Actions
        "running", "walking", "jumping", "swimming", "flying", "climbing", "dancing", "singing",
        "reading", "writing", "thinking", "learning", "teaching", "creating", "destroying", "building",
        # Time
        "past", "present", "future", "morning", "afternoon", "evening", "night", "day",
        "week", "month", "year", "century", "moment", "instant", "eternity", "forever",
        # Places
        "city", "village", "town", "country", "continent", "island", "peninsula", "coast",
        "valley", "plain", "plateau", "canyon", "cave", "lake", "pond", "stream"
    ]
    
    # Ensure we have exactly num_concepts
    if len(concepts) < num_concepts:
        # Add more generic concepts if needed
        for i in range(len(concepts), num_concepts):
            concepts.append(f"concept_{i+1}")
    concepts = concepts[:num_concepts]
    
    # Templates for generating text variations
    templates = [
        "This is about {concept}.",
        "The concept of {concept} is important.",
        "{concept} represents a fundamental idea.",
        "Understanding {concept} requires careful thought.",
        "Many people think about {concept} differently.",
        "The meaning of {concept} can vary across cultures."
    ]
    
    # Generate all combinations
    data = []
    for concept in concepts:
        for template_id, template in enumerate(templates):
            text = template.format(concept=concept)
            data.append({
                'concept': concept,
                'template_id': template_id,
                'template': template,
                'text': text
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} texts from {len(concepts)} concepts and {len(templates)} templates")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data()
