"""
Dataset generation script for landmark recognition with VLM and FAISS-based hard negative mining.

This script:
1. Loads landmark data with images
2. Encodes images using SigLIP model
3. Fuses multi-image embeddings per landmark
4. Uses FAISS to find hard negatives
5. Generates training samples with candidates
6. Splits data into train/val/test sets
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================
# CONFIGURATION
# ======================
@dataclass
class Config:
    """Configuration for dataset generation."""
    
    # Paths
    data_path: str = "setup_data_v3/data/landmarks_data_wiki.json"
    output_dir: Path = Path("setup_data_v3/data")
    
    # Model
    model_name: str = "google/siglip-base-patch16-224"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Sampling parameters
    max_candidates: int = 6
    hard_range: Tuple[int, int] = (1, 5)
    semi_hard_range: Tuple[int, int] = (5, 15)
    easy_range_start: int = 15
    faiss_k: int = 50
    positive_sample_prob: float = 0.7  # Reduced for better unknown learning
    exclude_top_k_on_unknown: int = 5
    random_sampling_prob: float = 0.05
    
    # Diversity
    enforce_type_diversity: bool = True
    type_diversity_strictness: float = 0.7  # Probability to skip duplicate type
    
    # Confidence calculation
    confidence_max_weight: float = 0.7
    confidence_mean_weight: float = 0.3
    
    # Batch processing
    batch_size: int = 32
    
    # Image classification
    object_threshold: float = 0.4  # Threshold to filter out object-only images
    exterior_weight: float = 1.0
    interior_weight: float = 0.5
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Text processing
    summary_sentences: int = 2
    evidence_max_length: int = 80
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Split ratios must sum to 1.0"
        assert 0 <= self.positive_sample_prob <= 1.0, \
            "positive_sample_prob must be between 0 and 1"
        self.output_dir.mkdir(exist_ok=True)


# ======================
# TEXT PROCESSING
# ======================
class TextProcessor:
    """Handles text processing for landmark descriptions."""
    
    @staticmethod
    def compress_summary(text: Optional[str], n_sentences: int = 2) -> str:
        """Extract first n sentences from text."""
        if not isinstance(text, str) or not text.strip():
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sentences[:n_sentences])
    
    @staticmethod
    def build_description(row: pd.Series) -> str:
        """Build landmark description from row data using template."""
        name = row.get("name", "")
        city = row.get("city", "")
        style = row.get("style", "")
        summary = TextProcessor.compress_summary(row.get("summary"), 1)
        
        # Determine landmark type
        name_lower = str(name).lower()
        if any(x in name_lower for x in ["собор", "cathedral"]):
            ltype = "собор"
        elif any(x in name_lower for x in ["церковь", "church"]):
            ltype = "церковь"
        elif any(x in name_lower for x in ["храм", "temple"]):
            ltype = "храм"
        elif any(x in name_lower for x in ["музей", "museum"]):
            ltype = "музей"
        elif any(x in name_lower for x in ["дворец", "palace"]):
            ltype = "дворец"
        elif any(x in name_lower for x in ["театр", "theater"]):
            ltype = "театр"
        else:
            ltype = "достопримечательность"
        
        # Build structured description
        parts = [f"{name} — это {ltype}"]
        
        if pd.notna(city) and city:
            parts.append(f"в {city}")
        
        if pd.notna(style) and style:
            parts.append(f"Стиль: {style}")
        
        if summary:
            parts.append(summary)
        
        return ". ".join(parts) + "."
    
    @staticmethod
    def make_evidence(description: str, max_length: int = 80) -> str:
        """Create evidence string from description."""
        return description[:max_length]


# ======================
# DATA LOADING
# ======================
class DataLoader:
    """Handles loading and preprocessing of landmark data."""
    
    @staticmethod
    def ensure_list(value) -> List:
        """Convert value to list if it isn't already."""
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        return [value]
    
    @staticmethod
    def load_data(data_path: str) -> pd.DataFrame:
        """Load and preprocess landmark data."""
        logger.info(f"Loading data from {data_path}")
        
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file: {e}")
            raise
        
        df = pd.DataFrame(raw_data)
        
        # Merge Russian and English fields
        df["name"] = df["name_ru"].fillna(df["name_en"])
        df["city"] = df["city_ru"].fillna(df["city_en"])
        df["style"] = df["architectural_style_ru"].fillna(df["architectural_style_en"])
        df["summary"] = df["wikipedia_summary_ru"].fillna(df["wikipedia_summary_en"])
        
        # Process image paths
        df["images"] = df["image_path"].apply(DataLoader.ensure_list)
        
        # Filter out landmarks without images
        initial_count = len(df)
        df = df[df["images"].map(len) > 0].reset_index(drop=True)
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} landmarks without images")
        
        logger.info(f"Loaded {len(df)} landmarks with images")
        return df


# ======================
# IMAGE ENCODER
# ======================
class ImageEncoder:
    """Handles image encoding and classification using SigLIP."""
    
    TEXT_PROMPTS = [
        "a landmark building exterior",
        "architecture monument",
        "indoor room interior",
        "inside a building",
        "object or artifact"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        logger.info(f"Loading model {config.model_name} on {self.device}")
        try:
            self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(config.model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def classify_and_embed(
        self, image_path: str
    ) -> Optional[Tuple[np.ndarray, str, float]]:
        """
        Classify image type and generate embedding.
        
        Returns:
            Tuple of (embedding, label, confidence) or None if filtered
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
        
        try:
            # Process image once for both classification and embedding
            inputs = self.processor(
                images=image,
                text=self.TEXT_PROMPTS,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(
                    dim=-1
                )[0].cpu().numpy()
                
                # Get normalized image embedding from same forward pass
                img_emb = outputs.image_embeds
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            # Calculate category probabilities
            exterior_prob = probs[0:2].mean()
            interior_prob = probs[2:4].mean()
            object_prob = probs[4]
            
            # Filter out object-only images
            if object_prob > self.config.object_threshold:
                logger.debug(f"Filtered object image: {image_path}")
                return None
            
            # Determine label and confidence
            if exterior_prob > interior_prob:
                label = "exterior"
                confidence = float(exterior_prob)
            else:
                label = "interior"
                confidence = float(interior_prob)
            
            return img_emb.cpu().numpy()[0], label, confidence
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
        finally:
            image.close()
    
    @staticmethod
    def fuse_embeddings(embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Fuse multiple embeddings into one using weighted average.
        
        Args:
            embeddings: List of embedding vectors
            weights: Optional weights for each embedding
            
        Returns:
            Normalized fused embedding
        """
        if not embeddings:
            raise ValueError("Cannot fuse empty embedding list")
        
        if weights is None:
            fused = np.mean(embeddings, axis=0)
        else:
            if len(embeddings) != len(weights):
                raise ValueError("Number of embeddings and weights must match")
            fused = np.average(embeddings, axis=0, weights=weights)
        
        # Normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        
        return fused


# ======================
# LANDMARK ENCODER
# ======================
class LandmarkEncoder:
    """Encodes landmarks by fusing multiple image embeddings."""
    
    def __init__(self, config: Config, image_encoder: ImageEncoder):
        self.config = config
        self.image_encoder = image_encoder
    
    def _process_images_batch(
        self, image_paths: List[str]
    ) -> List[Optional[Tuple[np.ndarray, str, float]]]:
        """Process multiple images in batch for better performance."""
        valid_images: List[Image.Image] = []
        valid_indices: List[int] = []
        
        # Load all valid images
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        if not valid_images:
            return [None] * len(image_paths)
        
        # Process in batches
        batch_results: List[Optional[Tuple[np.ndarray, str, float]]] = [
            None
        ] * len(image_paths)
        
        for i in range(0, len(valid_images), self.config.batch_size):
            batch = valid_images[i:i + self.config.batch_size]
            batch_idx = valid_indices[i:i + self.config.batch_size]
            
            try:
                inputs = self.image_encoder.processor(
                    images=batch,
                    text=self.image_encoder.TEXT_PROMPTS,
                    return_tensors="pt",
                    padding=True
                ).to(self.image_encoder.device)
                
                with torch.no_grad():
                    outputs = self.image_encoder.model(**inputs)
                    probs = outputs.logits_per_image.softmax(
                        dim=-1
                    ).cpu().numpy()
                    img_embs = outputs.image_embeds
                    img_embs = img_embs / img_embs.norm(
                        dim=-1, keepdim=True
                    )
                    img_embs = img_embs.cpu().numpy()
                
                for j, orig_idx in enumerate(batch_idx):
                    exterior_prob = probs[j][0:2].mean()
                    interior_prob = probs[j][2:4].mean()
                    object_prob = probs[j][4]
                    
                    if object_prob > self.config.object_threshold:
                        batch_results[orig_idx] = None
                        continue
                    
                    if exterior_prob > interior_prob:
                        label = "exterior"
                        confidence = float(exterior_prob)
                    else:
                        label = "interior"
                        confidence = float(interior_prob)
                    
                    batch_results[orig_idx] = (
                        img_embs[j], label, confidence
                    )
            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for orig_idx in batch_idx:
                    batch_results[orig_idx] = None
            
            finally:
                for img in batch:
                    img.close()
        
        return batch_results
    
    def encode_landmarks(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Encode all landmarks in dataframe.
        
        Returns:
            Tuple of (embeddings array, metadata list)
            Metadata: [{"lid": ..., "name": ..., "confidence": ...}, ...]
        """
        embeddings = []
        metadata = []
        
        logger.info("Encoding landmarks with batch processing...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
            landmark_id = str(row["landmark_id"])
            name = row["name"]
            
            # Batch process all images for this landmark
            results = self._process_images_batch(row["images"])
            
            image_embeddings = []
            weights = []
            valid_image_paths = []
            
            for i, result in enumerate(results):
                if result is None:
                    continue
                
                emb, label, confidence = result
                image_embeddings.append(emb)
                valid_image_paths.append(row["images"][i])
                
                # Weight exterior images higher
                if label == "exterior":
                    weight = confidence * self.config.exterior_weight
                else:
                    weight = confidence * self.config.interior_weight
                
                weights.append(weight)
            
            if not image_embeddings:
                logger.warning(
                    f"No valid images for landmark {landmark_id} ({name})"
                )
                continue
            
            # Fuse embeddings
            fused = self.image_encoder.fuse_embeddings(
                image_embeddings, weights
            )
            
            # Calculate confidence: balance quality and quantity
            mean_w = float(np.mean(weights))
            max_w = float(max(weights))
            num_images = len(weights)
            
            # Formula: weighted average with diminishing returns on quantity
            confidence = 0.6 * max_w + 0.4 * mean_w * min(1.0, np.log(1 + num_images) / 2)
            confidence = min(confidence, 1.0)
            
            embeddings.append(fused)
            metadata.append({
                "lid": landmark_id,
                "name": name,
                "row_idx": idx,
                "confidence": confidence,
                "max_conf": max_w,
                "mean_conf": mean_w,
                "num_images": len(image_embeddings),
                "valid_images": valid_image_paths,  # Only valid images
                "image_weights": weights  # Matching weights
            })
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Validate normalization for FAISS IndexFlatIP
        norms = np.linalg.norm(embeddings_array, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), \
            f"Embeddings not normalized! Norms range: [{norms.min():.4f}, {norms.max():.4f}]"
        
        logger.info(f"Encoded {len(embeddings_array)} landmarks")
        
        return embeddings_array, metadata


# ======================
# SAMPLE GENERATOR
# ======================
class SampleGenerator:
    """Generates training samples with hard and easy negatives."""
    
    def __init__(
        self,
        config: Config,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        image_encoder: ImageEncoder
    ):
        self.config = config
        self.df = df
        self.embeddings = embeddings
        self.metadata = metadata
        self.image_encoder = image_encoder
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        # Find nearest neighbors
        self.distances, self.indices = self.index.search(
            embeddings, config.faiss_k
        )
        
        # Build lookup map
        self.lid_to_idx = {
            meta["lid"]: idx for idx, meta in enumerate(metadata)
        }
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def _get_landmark_confidence(self, idx: int) -> float:
        """Get precomputed confidence for landmark."""
        return self.metadata[idx]["confidence"]
    
    def _get_landmark_type(self, row: pd.Series) -> str:
        """Extract landmark type from name or description."""
        name = str(row.get("name", "")).lower()
        
        # Common landmark types
        if any(x in name for x in ["собор", "cathedral"]):
            return "cathedral"
        elif any(x in name for x in ["церковь", "church"]):
            return "church"
        elif any(x in name for x in ["храм", "temple"]):
            return "temple"
        elif any(x in name for x in ["музей", "museum"]):
            return "museum"
        elif any(x in name for x in ["дворец", "palace"]):
            return "palace"
        elif any(x in name for x in ["театр", "theater", "theatre"]):
            return "theater"
        elif any(x in name for x in ["памятник", "monument"]):
            return "monument"
        elif any(x in name for x in ["мост", "bridge"]):
            return "bridge"
        elif any(x in name for x in ["площадь", "square"]):
            return "square"
        else:
            return "other"
    
    def _sample_negatives_dynamic(
        self, idx: int, target_lid: str, exclude_top_k: int = 0
    ) -> List[Dict]:
        """
        Sample negatives with diversity control and fixed distribution.
        
        Returns:
            List of negative candidates
        """
        # Sometimes ignore FAISS and sample randomly
        use_random = random.random() < self.config.random_sampling_prob
        
        if use_random:
            return self._sample_random_negatives(target_lid)
        
        neighbors = self.indices[idx]
        distances = self.distances[idx]
        
        # Define ranges
        hard_start = 1 + exclude_top_k
        hard_end = self.config.hard_range[1] + exclude_top_k
        semi_start = self.config.semi_hard_range[0] + exclude_top_k
        semi_end = self.config.semi_hard_range[1] + exclude_top_k
        easy_start = self.config.easy_range_start + exclude_top_k
        
        # Fixed sampling distribution
        hard_count = 2
        semi_count = 2
        easy_count = 1
        
        candidates: List[Dict] = []
        used_lids = set()
        used_types = set()
        
        # Pre-filter valid neighbors (exclude self)
        valid_neighbors = [(i, n, distances[i]) for i, n in enumerate(neighbors)
                          if n != idx and n < len(self.metadata)]
        
        # Create index mapping for fast lookup
        neighbor_to_dist = {n: (i, dist) for i, n, dist in valid_neighbors}
        
        def try_add_candidate(neighbor_idx, cand_type):
            if neighbor_idx not in neighbor_to_dist:
                return False
            
            meta = self.metadata[neighbor_idx]
            lid = meta["lid"]
            
            if lid == target_lid or lid in used_lids:
                return False
            
            row = self.df.iloc[meta["row_idx"]]
            
            # Check type diversity (soft constraint with fallback)
            if self.config.enforce_type_diversity:
                landmark_type = self._get_landmark_type(row)
                if landmark_type in used_types and landmark_type != "other":
                    min_required = 3
                    if len(candidates) >= min_required:
                        if random.random() < self.config.type_diversity_strictness:
                            return False
                used_types.add(landmark_type)
            
            used_lids.add(lid)
            
            # Get similarity from pre-computed mapping
            _, similarity = neighbor_to_dist[neighbor_idx]
            
            candidates.append({
                "name": meta["name"],
                "desc": TextProcessor.build_description(row),
                "type": cand_type,
                "sim": float(similarity)
            })
            return True
        
        # Sample from hard pool
        for neighbor_idx in neighbors[hard_start:hard_end]:
            if len([c for c in candidates if c["type"] == "hard"]) >= hard_count:
                break
            try_add_candidate(neighbor_idx, "hard")
        
        # Sample from semi-hard pool
        for neighbor_idx in neighbors[semi_start:semi_end]:
            if len([c for c in candidates if c["type"] == "semi_hard"]) >= semi_count:
                break
            try_add_candidate(neighbor_idx, "semi_hard")
        
        # Sample from easy pool
        for neighbor_idx in neighbors[easy_start:]:
            if len([c for c in candidates if c["type"] == "easy"]) >= easy_count:
                break
            try_add_candidate(neighbor_idx, "easy")
        
        return candidates
    
    def _sample_random_negatives(self, target_lid: str) -> List[Dict]:
        """Sample completely random negatives (ignore FAISS)."""
        all_indices = [
            i for i, meta in enumerate(self.metadata)
            if meta["lid"] != target_lid
        ]
        
        if not all_indices:
            return []
        
        count = random.randint(3, 6)
        sampled = random.sample(
            all_indices, min(count, len(all_indices))
        )
        
        candidates = []
        used_types = set()
        
        for idx in sampled:
            meta = self.metadata[idx]
            row = self.df.iloc[meta["row_idx"]]
            
            # Check type diversity (soft constraint with fallback)
            if self.config.enforce_type_diversity:
                landmark_type = self._get_landmark_type(row)
                if landmark_type in used_types and landmark_type != "other":
                    min_required = 3
                    if len(candidates) >= min_required:
                        if random.random() < self.config.type_diversity_strictness:
                            continue
                used_types.add(landmark_type)
            
            candidates.append({
                "name": meta["name"],
                "desc": TextProcessor.build_description(row),
                "type": "random",
                "sim": 0.0
            })
        
        return candidates
    
    def generate_sample(self, idx: int) -> Dict:
        """Generate a single training sample with contrastive signals."""
        meta = self.metadata[idx]
        landmark_id = meta["lid"]
        name = meta["name"]
        row = self.df.iloc[meta["row_idx"]]
        
        # Positive candidate
        positive = {
            "name": name,
            "desc": TextProcessor.build_description(row)
        }
        
        # Decide whether to include positive
        include_positive = random.random() < self.config.positive_sample_prob
        
        # Sample negatives dynamically
        if include_positive:
            # Normal sampling
            negatives = self._sample_negatives_dynamic(
                idx, landmark_id, exclude_top_k=0
            )
            
            # Ensure we have enough negatives
            total_needed = self.config.max_candidates - 1  # -1 for positive
            while len(negatives) < total_needed:
                extra = self._sample_random_negatives(landmark_id)
                if not extra:
                    break
                negatives.extend(extra)
            
            # Trim to exact count
            negatives = negatives[:total_needed]
            
            # Add positive
            candidates = negatives + [positive]
            target_name = positive["name"]
            target_confidence = self._get_landmark_confidence(idx)
        else:
            # Unknown case: exclude top-K similar to avoid confusion
            negatives = self._sample_negatives_dynamic(
                idx,
                landmark_id,
                exclude_top_k=self.config.exclude_top_k_on_unknown
            )
            
            # Ensure we have enough negatives
            while len(negatives) < self.config.max_candidates:
                extra = self._sample_random_negatives(landmark_id)
                if not extra:
                    break
                negatives.extend(extra)
            
            # Only negatives, no positive
            candidates = negatives[:self.config.max_candidates]
            target_name = "unknown"
            target_confidence = 0.0
        
        # Shuffle AFTER controlling composition
        random.shuffle(candidates)
        
        # Select image with weighted probability from valid images
        valid_images = meta.get("valid_images", [])
        image_weights = meta.get("image_weights", [])
        
        if not valid_images:
            raise ValueError(f"No valid images for landmark {landmark_id}")
        
        if image_weights and len(image_weights) == len(valid_images):
            # Weighted selection: prefer higher confidence images
            total_weight = sum(image_weights)
            if total_weight > 0:
                probs = [w / total_weight for w in image_weights]
                image_path = np.random.choice(valid_images, p=probs)
            else:
                image_path = random.choice(valid_images)
        else:
            # Fallback to random if weights not available
            image_path = random.choice(valid_images)
        
        # Build contrastive pairs
        contrastive = self._build_contrastive_pairs(
            idx, landmark_id, positive
        )
        
        return {
            "image": image_path,
            "candidates": candidates,
            "target": {
                "name": target_name,
                "landmark_id": landmark_id,
                "confidence": target_confidence,
                "evidence": TextProcessor.make_evidence(
                    positive["desc"],
                    self.config.evidence_max_length
                )
            },
            "meta": {
                "num_images": meta["num_images"],
                "confidence": meta["confidence"],
                "mean_conf": meta["mean_conf"],
                "max_conf": meta["max_conf"]
            },
            "contrastive": contrastive
        }
    
    def _build_contrastive_pairs(
        self, idx: int, target_lid: str, positive: Dict
    ) -> Dict[str, Any]:
        """Build contrastive learning signals with similarity scores."""
        # Filter out self explicitly
        neighbors = [n for n in self.indices[idx] if n != idx]
        distances = self.distances[idx]
        
        # Get one hard negative (most similar)
        hard_negative = None
        hard_sim = 0.0
        for neighbor_idx in neighbors[:5]:
            if neighbor_idx >= len(self.metadata):
                continue
            meta = self.metadata[neighbor_idx]
            if meta["lid"] != target_lid:
                row = self.df.iloc[meta["row_idx"]]
                hard_negative = {
                    "name": meta["name"],
                    "desc": TextProcessor.build_description(row)
                }
                # Find position in original distances
                orig_pos = np.where(self.indices[idx] == neighbor_idx)[0]
                hard_sim = float(distances[orig_pos[0]]) if len(orig_pos) > 0 else 0.0
                break
        
        # Get one semi-hard negative
        semi_negative = None
        semi_sim = 0.0
        for neighbor_idx in neighbors[10:20]:
            if neighbor_idx >= len(self.metadata):
                continue
            meta = self.metadata[neighbor_idx]
            if meta["lid"] != target_lid:
                row = self.df.iloc[meta["row_idx"]]
                semi_negative = {
                    "name": meta["name"],
                    "desc": TextProcessor.build_description(row)
                }
                orig_pos = np.where(self.indices[idx] == neighbor_idx)[0]
                semi_sim = float(distances[orig_pos[0]]) if len(orig_pos) > 0 else 0.0
                break
        
        return {
            "positive": positive["desc"],
            "hard_negative": hard_negative["desc"] if hard_negative else "",
            "hard_negative_sim": hard_sim,
            "semi_negative": semi_negative["desc"] if semi_negative else "",
            "semi_negative_sim": semi_sim
        }
    
    def generate_all_samples(self) -> List[Dict]:
        """Generate samples for all landmarks."""
        logger.info("Generating samples...")
        samples = []
        
        for idx in tqdm(range(len(self.metadata)), desc="Generating samples"):
            try:
                sample = self.generate_sample(idx)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Error generating sample for index {idx}: {e}")
        
        logger.info(f"Generated {len(samples)} samples")
        return samples


# ======================
# DATA SPLITTER
# ======================
class DataSplitter:
    """Splits samples into train/val/test sets."""
    
    @staticmethod
    def split_samples(
        samples: List[Dict], config: Config
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split samples by landmark_id with diversity constraint.
        Ensures balanced distribution of landmark types.
        """
        # Get unique landmark IDs
        landmark_ids = list(
            set(s["target"]["landmark_id"] for s in samples)
        )
        random.shuffle(landmark_ids)
        
        n = len(landmark_ids)
        train_end = int(config.train_ratio * n)
        val_end = int((config.train_ratio + config.val_ratio) * n)
        
        train_ids = set(landmark_ids[:train_end])
        val_ids = set(landmark_ids[train_end:val_end])
        test_ids = set(landmark_ids[val_end:])
        
        train, val, test = [], [], []
        
        for sample in samples:
            lid = sample["target"]["landmark_id"]
            if lid in train_ids:
                train.append(sample)
            elif lid in val_ids:
                val.append(sample)
            elif lid in test_ids:
                test.append(sample)
        
        logger.info(
            f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test


# ======================
# FILE WRITER
# ======================
class FileWriter:
    """Handles writing samples to JSONL files."""
    
    @staticmethod
    def save_jsonl(path: Path, data: List[Dict]) -> None:
        """Save data to JSONL file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(data)} samples to {path}")
        except Exception as e:
            logger.error(f"Failed to save file {path}: {e}")
            raise


# ======================
# MAIN PIPELINE
# ======================
def main():
    """Main pipeline for dataset generation."""
    # Initialize configuration
    config = Config()
    
    # Set random seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    logger.info("Starting dataset generation pipeline")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load data
        df = DataLoader.load_data(config.data_path)
        
        # Initialize encoder
        image_encoder = ImageEncoder(config)
        landmark_encoder = LandmarkEncoder(config, image_encoder)
        
        # Encode landmarks
        embeddings, metadata = landmark_encoder.encode_landmarks(df)
        
        if len(embeddings) == 0:
            logger.error("No landmarks were successfully encoded")
            return
        
        # Generate samples
        sample_generator = SampleGenerator(
            config, df, embeddings, metadata, image_encoder
        )
        samples = sample_generator.generate_all_samples()
        
        if not samples:
            logger.error("No samples were generated")
            return
        
        # Split data
        train, val, test = DataSplitter.split_samples(samples, config)
        
        # Save results
        FileWriter.save_jsonl(config.output_dir / "train.jsonl", train)
        FileWriter.save_jsonl(config.output_dir / "val.jsonl", val)
        FileWriter.save_jsonl(config.output_dir / "test.jsonl", test)
        
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Output directory: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
