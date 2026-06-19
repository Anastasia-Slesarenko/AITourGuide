"""
Step 6: Dataset generation script for landmark recognition with FAISS-based
hard negative mining.

This script inherits from src.rag.landmark_retriever and adds training-specific
logic:
- Hard negative mining
- Label generation (target_idx)
- Train/val/test splitting
- Sample generation with unknown ratio

Prerequisites:
    - Requires landmarks_with_guide_descriptions.json from step5
    - Input file contains validated images with captions

This script:
1. Loads landmark data with validated images
2. Splits images into gallery/query roles (production-correct)
3. Builds FAISS index from gallery images only
4. Generates training samples with hard negatives
5. Saves samples and training index for reuse

Training Index Reuse:
    The script saves a training gallery index (gallery images only) that can be
    reused for experiments with different parameters:
    
    - training_gallery_index.faiss: FAISS index
    - training_gallery_metadata.json: Image metadata
    - training_gallery_embeddings.npy: SigLIP embeddings
    
    To reuse existing index (default):
        config.reuse_training_index = True
        config.force_rebuild_index = False
    
    To force rebuild:
        config.force_rebuild_index = True
    
    NOTE: This is NOT the production index. For production, use
    LandmarkRetriever.build_index_from_landmarks() which includes ALL images.

Usage:
    python step6_setup_dataset.py
"""

from __future__ import annotations

# ---------------------------------------------------------------
# macOS segfault prevention: ALL thread/process env vars MUST be
# set before any native library (numpy, torch, faiss) is imported.
# ---------------------------------------------------------------
import os
import platform as _platform

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if _platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ---------------------------------------------------------------

import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from production modules
from src.rag.indexing_v2 import IndexBuilder, IndexConfig
from src.rag.landmark_retriever import (
    GalleryImageMetadata,
    ScoreAggregator,
    build_index_from_landmarks
)

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
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Paths
    data_path: str = "data/processed/landmarks_with_guide_descriptions_filtred.json"
    output_dir: Path = Path("data/processed/")
    image_base_dir: Path = Path("images")
    
    # Embedder backend: "siglip" or "dinov2"
    # "siglip" uses google/siglip-so400m-patch14-384 with exterior/interior
    #          classification and object filtering.
    # "dinov2" uses facebook/dinov2-base without any classification step —
    #          all images are encoded with uniform weight.
    embedder_type: str = "siglip"
    embedder_model: str = ""  # Leave empty to use the default for each type
    
    # Index reuse
    reuse_training_index: bool = True  # Load existing training index if available
    force_rebuild_index: bool = False  # Force rebuild even if index exists
    
    # Sampling parameters
    max_candidates: int = 15
    faiss_k: int = 100
    max_gallery_per_landmark: int = 5
    
    # Diversity
    enforce_type_diversity: bool = True
    type_diversity_strictness: float = 0.7
    min_candidates_before_diversity: int = 3
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Sparse landmark split ratios (for 2-3 image landmarks)
    two_image_train_ratio: float = 0.8
    two_image_val_ratio: float = 0.1
    two_image_test_ratio: float = 0.1
    
    three_image_train_ratio: float = 0.7
    three_image_val_ratio: float = 0.15
    three_image_test_ratio: float = 0.15
    
    # Encoding batch size
    encoding_batch_size: int = 32
    
    # Gallery augmentation (applied during gallery index building)
    # Augmented versions increase gallery coverage for sparse landmarks
    gallery_augmentations: int = 5  # N augmented versions per gallery image
    gallery_augmentation_threshold: int = 5  # Augment if landmark has < N images
    
    # Unknown sample ratio
    unknown_ratio: float = 0.2

    # Hard unknown sample ratio (unknown samples with high retrieval scores).
    # These are the hardest cases: all candidates look visually similar to the
    # query (retrieval_score >= hard_unknown_min_score) but the correct answer
    # is NOT in the candidate list.
    # Simulates production scenario where retrieval is confident but wrong.
    # Set to 0.0 to disable (backward-compatible default).
    hard_unknown_ratio: float = 0.08  # Было 0.15 в v2
    # Minimum retrieval score for hard unknown candidates.
    # Candidates below this threshold are filtered out.
    hard_unknown_min_score: float = 0.88  # Было 0.85 — только самые сложные

    # Score aggregation strategy
    score_aggregation_mode: str = "weighted_top2" # "max" "weighted_top2"
    weighted_top2_alpha: float = 0.8  # Weight for top score
    
    # Candidate image selection
    candidate_image_selection_mode: str = "top1"  # "top1" or "sample_topk"
    candidate_image_sample_topk: int = 3  # Used if mode is "sample_topk"
    
    # Unknown sample generation
    unknown_generation_strategy: str = "outside_topk"
    unknown_exclude_topk: int = 25  # Exclude top-k from unknown candidates
    
    # Candidate hardness classification
    hardness_classification_mode: str = "score_based"
    hard_threshold: float = 0.85  # score >= 0.85 -> hard
    semi_hard_threshold: float = 0.75  # 0.75 <= score < 0.85 -> semi-hard
    # score < 0.75 -> easy
    
    # Text processing
    evidence_max_length: int = 80
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        
        # Validate sparse landmark ratios
        assert abs(self.two_image_train_ratio + self.two_image_val_ratio +
                   self.two_image_test_ratio - 1.0) < 1e-6, \
            "2-image ratios must sum to 1.0"
        assert abs(self.three_image_train_ratio + self.three_image_val_ratio +
                   self.three_image_test_ratio - 1.0) < 1e-6, \
            "3-image ratios must sum to 1.0"
        
        # Validate score aggregation
        valid_modes = ["max", "top2_mean", "weighted_top2"]
        assert self.score_aggregation_mode in valid_modes, \
            f"score_aggregation_mode must be one of {valid_modes}"
        assert 0 <= self.weighted_top2_alpha <= 1.0, \
            "weighted_top2_alpha must be between 0 and 1"
        
        # Validate candidate selection
        assert self.candidate_image_sample_topk >= 1, \
            "candidate_image_sample_topk must be >= 1"
        
        # Validate unknown generation
        valid_strategies = ["outside_topk", "manual_removal"]
        assert self.unknown_generation_strategy in valid_strategies, \
            f"unknown_generation_strategy must be one of {valid_strategies}"
        assert self.unknown_exclude_topk >= 1, \
            "unknown_exclude_topk must be >= 1"
        
        # Validate hardness classification
        valid_modes = ["score_based", "rank_based"]
        assert self.hardness_classification_mode in valid_modes, \
            f"hardness_classification_mode must be one of {valid_modes}"
        assert 0 <= self.semi_hard_threshold <= self.hard_threshold <= 1.0, \
            "Thresholds must satisfy: 0 <= semi_hard <= hard <= 1.0"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ======================
# TEXT PROCESSING
# ======================
class TextProcessor:
    """Handles text processing for landmark descriptions."""
    
    @staticmethod
    def get_landmark_summary_caption(row: pd.Series) -> str:
        """Get landmark_summary_caption from row."""
        caption = row.get("landmark_summary_caption", "")
        if caption and isinstance(caption, str) and caption.strip():
            return str(caption)
        return str(row.get("name", ""))
    
    @staticmethod
    def get_image_and_caption_for_candidate(
        row: pd.Series, valid_images: List[str]
    ) -> Tuple[str, str, str]:
        """Get random image and caption for a candidate landmark.
        
        Returns:
            Tuple of (candidate_image, caption, caption_landmark)
        """
        if not valid_images:
            return "", str(row.get("name", "")), str(row.get("name", ""))
        
        candidate_image = random.choice(valid_images)
        caption_landmark = TextProcessor.get_landmark_summary_caption(row)
        
        # Get specific caption for the selected image
        valid_images_data = row.get("valid_images", [])
        caption = str(row.get("name", ""))  # Default fallback
        
        for img_data in valid_images_data:
            if isinstance(img_data, dict):
                img_path = img_data.get("path", "")
                if img_path == candidate_image:
                    img_caption = img_data.get("caption", "")
                    if img_caption and isinstance(img_caption, str) and img_caption.strip():
                        caption = str(img_caption)
                    break
        
        return candidate_image, caption, caption_landmark
    
    @staticmethod
    def make_evidence(description: str, max_length: int = 80) -> str:
        """Create evidence string from description."""
        return description[:max_length]


# ======================
# IMAGE ROLE ASSIGNMENT
# ======================
@dataclass
class LandmarkImageSplit:
    """
    Defines how images are split for a single landmark.
    
    This is the core data structure for production-correct closed-set
    retrieval. Each landmark's images are assigned specific roles:
    - gallery: used in FAISS index for retrieval
    - query_train/val/test: used as query images in respective splits
    
    Key principle: NO OVERLAP between gallery and query images.
    """
    landmark_id: str
    landmark_name: str
    total_images: int
    
    # Image role assignments
    gallery_images: List[str] = field(default_factory=list)
    query_train_images: List[str] = field(default_factory=list)
    query_val_images: List[str] = field(default_factory=list)
    query_test_images: List[str] = field(default_factory=list)
    
    # Capability flags
    retrieval_only: bool = False  # True if only 1 image
    supports_contrastive: bool = False  # True if >= 3 images
    supports_reranking: bool = True  # True if >= 2 images
    
    # Metadata (preserved from original)
    confidence: float = 0.0
    mean_conf: float = 0.0
    max_conf: float = 0.0
    guide_description: str = ""
    row_idx: int = -1
    
    def get_all_query_images(self) -> List[str]:
        """Get all query images across all splits."""
        return (
            self.query_train_images +
            self.query_val_images +
            self.query_test_images
        )
    
    def validate(self) -> None:
        """Validate no overlap between gallery and query images."""
        gallery_set = set(self.gallery_images)
        query_set = set(self.get_all_query_images())
        
        overlap = gallery_set & query_set
        if overlap:
            raise ValueError(
                f"Landmark {self.landmark_id}: "
                f"Gallery and query images overlap: {overlap}"
            )
        
        # Validate total count
        assigned = (
            len(self.gallery_images) +
            len(self.query_train_images) +
            len(self.query_val_images) +
            len(self.query_test_images)
        )
        if assigned != self.total_images:
            logger.warning(
                f"Landmark {self.landmark_id}: "
                f"Assigned {assigned} images but total is {self.total_images}"
            )


# ======================
# DATA SPLITTER (NEW)
# ======================
class ImageRoleSplitter:
    """
    Splits images within each landmark into gallery/query roles.
    
    This replaces the old DataSplitter which split by landmark_id.
    
    Key differences:
    - OLD: train_landmarks != val_landmarks (open-set)
    - NEW: all landmarks in all splits, but different query images
    
    Adaptive strategy based on image count:
    - 1 image: gallery only (retrieval_only=True)
    - 2 images: probabilistic split across train/val/test (80/10/10)
    - 3 images: probabilistic split across train/val/test (70/15/15)
    - 4+ images: N-3 gallery + 1 train + 1 val + 1 test
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        max_gallery_per_landmark: int = 5
    ):
        self.config = config
        self.max_gallery_per_landmark = max_gallery_per_landmark
        
        # Statistics
        self.stats = {
            "total_landmarks": 0,
            "retrieval_only": 0,  # 1 image
            "two_image_train": 0,  # 2 images -> train
            "two_image_val": 0,    # 2 images -> val
            "two_image_test": 0,   # 2 images -> test
            "three_image_train": 0,  # 3 images -> train
            "three_image_val": 0,    # 3 images -> val
            "three_image_test": 0,   # 3 images -> test
            "full_split": 0,  # 4+ images
            "total_gallery_images": 0,
            "total_query_train": 0,
            "total_query_val": 0,
            "total_query_test": 0,
        }
    
    def _split_images_for_landmark(
        self,
        images: List[str],
        landmark_id: str
    ) -> Dict[str, List[str]]:
        """
        Split images for a single landmark using adaptive strategy.
        
        Returns dict with keys: gallery, query_train, query_val, query_test
        """
        num_images = len(images)
        
        # Shuffle for randomness (but reproducible via seed)
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        if num_images == 1:
            # Edge case: only 1 image
            # Use for gallery only, no queries
            logger.debug(
                f"Landmark {landmark_id}: 1 image -> gallery only"
            )
            return {
                "gallery": shuffled,
                "query_train": [],
                "query_val": [],
                "query_test": []
            }
        
        elif num_images == 2:
            # NEW: Probabilistic split for 2-image landmarks
            # 80% -> gallery + train
            # 10% -> gallery + val
            # 10% -> gallery + test
            rand_val = random.random()
            
            if rand_val < self.config.two_image_train_ratio:
                # Train split
                split_type = "train"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [shuffled[1]],
                    "query_val": [],
                    "query_test": []
                }
            elif rand_val < (self.config.two_image_train_ratio +
                            self.config.two_image_val_ratio):
                # Val split
                split_type = "val"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [],
                    "query_val": [shuffled[1]],
                    "query_test": []
                }
            else:
                # Test split
                split_type = "test"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [],
                    "query_val": [],
                    "query_test": [shuffled[1]]
                }
            
            logger.debug(
                f"Landmark {landmark_id}: 2 images -> "
                f"1 gallery + 1 {split_type}"
            )
            return result
        
        elif num_images == 3:
            # NEW: Probabilistic split for 3-image landmarks
            # 70% -> 2 gallery + train
            # 15% -> 2 gallery + val
            # 15% -> 2 gallery + test
            rand_val = random.random()
            
            if rand_val < self.config.three_image_train_ratio:
                # Train split
                split_type = "train"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [shuffled[2]],
                    "query_val": [],
                    "query_test": []
                }
            elif rand_val < (self.config.three_image_train_ratio +
                            self.config.three_image_val_ratio):
                # Val split
                split_type = "val"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [],
                    "query_val": [shuffled[2]],
                    "query_test": []
                }
            else:
                # Test split
                split_type = "test"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [],
                    "query_val": [],
                    "query_test": [shuffled[2]]
                }
            
            logger.debug(
                f"Landmark {landmark_id}: 3 images -> "
                f"2 gallery + 1 {split_type}"
            )
            return result
        
        else:
            # Normal case: 4+ images
            # Reserve 1 for train, 1 for val, 1 for test
            # Rest go to gallery (capped at max_gallery_per_landmark)
            
            # Reserve query images first
            query_test = [shuffled[-1]]
            query_val = [shuffled[-2]]
            query_train = [shuffled[-3]]
            
            # Remaining images for gallery
            gallery_candidates = shuffled[:-3]
            
            # Cap gallery size to prevent popular landmarks from dominating
            if len(gallery_candidates) > self.max_gallery_per_landmark:
                gallery = gallery_candidates[:self.max_gallery_per_landmark]
                logger.debug(
                    f"Landmark {landmark_id}: "
                    f"Capped gallery from {len(gallery_candidates)} "
                    f"to {self.max_gallery_per_landmark}"
                )
            else:
                gallery = gallery_candidates
            
            logger.debug(
                f"Landmark {landmark_id}: {num_images} images -> "
                f"{len(gallery)} gallery + 1 train + 1 val + 1 test"
            )
            
            return {
                "gallery": gallery,
                "query_train": query_train,
                "query_val": query_val,
                "query_test": query_test
            }
    
    def split_all_landmarks(
        self,
        df: pd.DataFrame
    ) -> List[LandmarkImageSplit]:
        """
        Split images for all landmarks in the dataframe.
        
        Args:
            df: DataFrame with columns: landmark_id, images, name, etc.
        
        Returns:
            List of LandmarkImageSplit objects, one per landmark
        """
        logger.info("Starting image role assignment for all landmarks...")
        
        splits = []
        
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Splitting landmark images"
        ):
            landmark_id = str(row.get("landmark_id", ""))
            name = str(row.get("name", ""))
            images = row.get("images", [])
            
            if not images:
                logger.warning(
                    f"Landmark {landmark_id} has no images, skipping"
                )
                continue
            
            # Apply adaptive split strategy
            split_result = self._split_images_for_landmark(
                images, landmark_id
            )
            
            # Create LandmarkImageSplit object
            split = LandmarkImageSplit(
                landmark_id=landmark_id,
                landmark_name=name,
                total_images=len(images),
                gallery_images=split_result["gallery"],
                query_train_images=split_result["query_train"],
                query_val_images=split_result["query_val"],
                query_test_images=split_result["query_test"],
                retrieval_only=(len(images) == 1),
                supports_contrastive=(len(images) >= 3),
                supports_reranking=(len(images) >= 2),
                confidence=row.get("confidence", 0.0),
                mean_conf=row.get("mean_conf", 0.0),
                max_conf=row.get("max_conf", 0.0),
                guide_description=row.get("guide_description", ""),
                row_idx=idx
            )
            
            # Validate
            split.validate()
            
            splits.append(split)
            
            # Update statistics
            self.stats["total_landmarks"] += 1
            self.stats["total_gallery_images"] += len(split.gallery_images)
            self.stats["total_query_train"] += len(
                split.query_train_images
            )
            self.stats["total_query_val"] += len(split.query_val_images)
            self.stats["total_query_test"] += len(split.query_test_images)
            
            # Update statistics based on actual split
            if len(images) == 1:
                self.stats["retrieval_only"] += 1
            elif len(images) == 2:
                if split.query_train_images:
                    self.stats["two_image_train"] += 1
                elif split.query_val_images:
                    self.stats["two_image_val"] += 1
                elif split.query_test_images:
                    self.stats["two_image_test"] += 1
            elif len(images) == 3:
                if split.query_train_images:
                    self.stats["three_image_train"] += 1
                elif split.query_val_images:
                    self.stats["three_image_val"] += 1
                elif split.query_test_images:
                    self.stats["three_image_test"] += 1
            else:
                self.stats["full_split"] += 1
        
        # Log statistics
        self._log_statistics()
        
        return splits
    
    def _log_statistics(self) -> None:
        """Log detailed statistics about the split."""
        logger.info("=" * 60)
        logger.info("IMAGE ROLE ASSIGNMENT STATISTICS")
        logger.info("=" * 60)
        logger.info(
            f"Total landmarks: {self.stats['total_landmarks']}"
        )
        logger.info(
            f"  - Retrieval only (1 img): "
            f"{self.stats['retrieval_only']}"
        )
        
        # 2-image landmarks breakdown
        two_img_total = (self.stats['two_image_train'] +
                        self.stats['two_image_val'] +
                        self.stats['two_image_test'])
        if two_img_total > 0:
            logger.info(f"  - 2 images: {two_img_total}")
            logger.info(
                f"    * Train: {self.stats['two_image_train']} "
                f"({100*self.stats['two_image_train']/two_img_total:.1f}%)"
            )
            logger.info(
                f"    * Val: {self.stats['two_image_val']} "
                f"({100*self.stats['two_image_val']/two_img_total:.1f}%)"
            )
            logger.info(
                f"    * Test: {self.stats['two_image_test']} "
                f"({100*self.stats['two_image_test']/two_img_total:.1f}%)"
            )
        
        # 3-image landmarks breakdown
        three_img_total = (self.stats['three_image_train'] +
                          self.stats['three_image_val'] +
                          self.stats['three_image_test'])
        if three_img_total > 0:
            logger.info(f"  - 3 images: {three_img_total}")
            logger.info(
                f"    * Train: {self.stats['three_image_train']} "
                f"({100*self.stats['three_image_train']/three_img_total:.1f}%)"
            )
            logger.info(
                f"    * Val: {self.stats['three_image_val']} "
                f"({100*self.stats['three_image_val']/three_img_total:.1f}%)"
            )
            logger.info(
                f"    * Test: {self.stats['three_image_test']} "
                f"({100*self.stats['three_image_test']/three_img_total:.1f}%)"
            )
        
        logger.info(
            f"  - Full split (4+ imgs): {self.stats['full_split']}"
        )
        logger.info("")
        logger.info(
            f"Gallery images: {self.stats['total_gallery_images']}"
        )
        logger.info(
            f"Query images (train): {self.stats['total_query_train']}"
        )
        logger.info(
            f"Query images (val): {self.stats['total_query_val']}"
        )
        logger.info(
            f"Query images (test): {self.stats['total_query_test']}"
        )
        logger.info("=" * 60)


# ======================
# GALLERY INDEX BUILDER
# ======================
# Note: GalleryImageMetadata is imported from src.rag.landmark_retriever

class GalleryIndexBuilder:
    """
    Builds image-level FAISS index from gallery images only.
    
    Key differences from old approach:
    - OLD: One embedding per landmark (mean pooling)
    - NEW: One embedding per gallery image
    
    - OLD: Index contains all images
    - NEW: Index contains ONLY gallery images
    
    - OLD: Retrieval returns landmarks
    - NEW: Retrieval returns images, then aggregate by landmark_id
    
    This matches production behavior where:
    1. Query image → SigLIP embedding
    2. FAISS search → top-k gallery images
    3. Group by landmark_id → max similarity per landmark
    4. Return top landmarks
    """
    
    def __init__(
        self,
        index_builder: IndexBuilder,
        batch_size: int = 32,
        gallery_augmentations: int = 4,
        gallery_augmentation_threshold: int = 4
    ):
        """
        Args:
            index_builder: IndexBuilder from src.rag.indexing_v2
            batch_size: Batch size for encoding images
            gallery_augmentations: N augmented versions per gallery image
                (0 = disabled). Applied only for landmarks with fewer
                images than gallery_augmentation_threshold.
            gallery_augmentation_threshold: Augment only if landmark
                has fewer gallery images than this value.
        """
        self.index_builder = index_builder
        self.batch_size = batch_size
        self.gallery_augmentations = gallery_augmentations
        self.gallery_augmentation_threshold = gallery_augmentation_threshold
        self.stats = {
            "total_landmarks": 0,
            "total_gallery_images": 0,
            "total_augmented_images": 0,
            "landmarks_with_1_image": 0,
            "landmarks_with_2_images": 0,
            "landmarks_with_3_images": 0,
            "landmarks_with_4plus_images": 0,
        }
    
    @staticmethod
    def encode_single_image(
        image_path: Path,
        encoder
    ) -> Optional[np.ndarray]:
        """
        Encode a single image with proper resource management.
        
        Args:
            image_path: Path to image file
            encoder: Encoder instance with encode_batch method
        
        Returns:
            Image embedding or None if encoding failed
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                results = encoder.encode_batch([img])
                
                if results and results[0] is not None:
                    embedding, _, _ = results[0]
                    return embedding
            return None
        except Exception as e:
            logger.error(f"Failed to encode {image_path}: {e}")
            return None
    
    def encode_images_batch(
        self,
        image_paths: List[Path]
    ) -> List[Optional[np.ndarray]]:
        """
        Encode multiple images in batches with proper resource management.
        
        Args:
            image_paths: List of image paths to encode
        
        Returns:
            List of embeddings (None for failed images)
        """
        from PIL import Image
        all_embeddings = []
        
        # Add progress bar for encoding
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(
            total=len(image_paths),
            desc="Encoding gallery images",
            unit="img"
        )
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            failed_indices = set()
            
            # Load batch images with proper resource management
            for idx, img_path in enumerate(batch_paths):
                try:
                    with Image.open(img_path) as raw_img:
                        batch_images.append(raw_img.convert("RGB"))
                except Exception as e:
                    logger.debug(f"Failed to load {img_path}: {e}")
                    failed_indices.add(idx)
                    batch_images.append(None)
            
            # Encode batch (only valid images)
            valid_images = [img for img in batch_images if img is not None]
            
            if valid_images:
                try:
                    results = self.index_builder.encoder.encode_batch(
                        valid_images
                    )
                    
                    # Map results back to original batch positions
                    result_idx = 0
                    for idx in range(len(batch_paths)):
                        if idx in failed_indices:
                            all_embeddings.append(None)
                        else:
                            if (results and result_idx < len(results) and
                                    results[result_idx] is not None):
                                embedding, _, _ = results[result_idx]
                                all_embeddings.append(embedding)
                            else:
                                all_embeddings.append(None)
                            result_idx += 1
                            
                except Exception as e:
                    logger.error(f"Batch encoding failed: {e}")
                    # Add None for all images in failed batch
                    all_embeddings.extend([None] * len(batch_paths))
            else:
                # All images in batch failed to load
                all_embeddings.extend([None] * len(batch_paths))
            
            # Close loaded images to prevent memory leak
            for img in batch_images:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
            
            # Update progress bar
            pbar.update(len(batch_paths))
        
        pbar.close()
        return all_embeddings
    
    def build_from_splits(
        self,
        splits: List[LandmarkImageSplit],
        df: pd.DataFrame,
        image_base_dir: Path
    ) -> Tuple[np.ndarray, List[GalleryImageMetadata], faiss.Index]:
        """
        Build FAISS index from gallery images in splits.
        
        Args:
            splits: List of LandmarkImageSplit objects
            df: Original dataframe with landmark data
            image_base_dir: Base directory for images
        
        Returns:
            embeddings: (N, D) array of gallery image embeddings
            metadata: List of GalleryImageMetadata, one per gallery image
            index: FAISS index containing gallery embeddings
        """
        logger.info("Building gallery index from image splits...")
        
        # Collect all gallery images with metadata
        gallery_data = []
        image_id = 0
        
        for split in splits:
            if not split.gallery_images:
                continue
            
            # Get row from dataframe
            row = df.iloc[split.row_idx]
            
            # Get caption_landmark for this landmark
            caption_landmark = TextProcessor.get_landmark_summary_caption(row)
            
            # Build image path to caption mapping from valid_images
            valid_images = row.get("valid_images", [])
            image_caption_map = {}
            for img_data in valid_images:
                if isinstance(img_data, dict):
                    img_path = img_data.get("path", "")
                    img_caption = img_data.get("caption", "")
                    if img_path:
                        image_caption_map[img_path] = img_caption if img_caption else row.get("name", "")
            
            for img_path in split.gallery_images:
                # Get specific caption for this image
                caption = image_caption_map.get(img_path, row.get("name", ""))
                
                gallery_data.append({
                    "image_id": image_id,
                    "image_path": img_path,
                    "landmark_id": split.landmark_id,
                    "landmark_name": split.landmark_name,
                    "caption": caption,
                    "caption_landmark": caption_landmark,
                    "confidence": split.confidence,
                    "mean_conf": split.mean_conf,
                    "max_conf": split.max_conf,
                    "guide_description": split.guide_description,
                    "row_idx": split.row_idx,
                    "num_gallery_images": len(split.gallery_images)
                })
                image_id += 1
            
            # Update statistics
            self.stats["total_landmarks"] += 1
            self.stats["total_gallery_images"] += len(split.gallery_images)
            
            num_imgs = len(split.gallery_images)
            if num_imgs == 1:
                self.stats["landmarks_with_1_image"] += 1
            elif num_imgs == 2:
                self.stats["landmarks_with_2_images"] += 1
            elif num_imgs == 3:
                self.stats["landmarks_with_3_images"] += 1
            else:
                self.stats["landmarks_with_4plus_images"] += 1
        
        logger.info(
            f"Collected {len(gallery_data)} gallery images "
            f"from {self.stats['total_landmarks']} landmarks"
        )
        
        # Encode gallery images using batching
        logger.info(
            f"Encoding gallery images in batches of {self.batch_size}..."
        )
        if self.gallery_augmentations > 0:
            logger.info(
                f"Gallery augmentation enabled: "
                f"{self.gallery_augmentations} augments per image "
                f"(threshold: < {self.gallery_augmentation_threshold} images)"
            )
        
        embeddings_list = []
        metadata_list = []
        
        # Prepare paths and validate existence
        valid_items = []
        image_paths = []
        
        for item in gallery_data:
            img_path = image_base_dir / item["image_path"]
            if img_path.exists():
                valid_items.append(item)
                image_paths.append(img_path)
            else:
                logger.warning(f"Image not found: {img_path}")
        
        logger.info(f"Found {len(image_paths)} valid images to encode")
        
        # Encode original gallery images in batches
        embeddings = self.encode_images_batch(image_paths)
        logger.info("Gallery image encoding complete. Processing results...")
        
        augment_fn = self.index_builder.encoder.augment_fn
        do_augment = (
            self.gallery_augmentations > 0
            and augment_fn is not None
        )
        
        if do_augment:
            aug_count = sum(
                1 for item in valid_items
                if item["num_gallery_images"] < self.gallery_augmentation_threshold
            )
            logger.info(
                f"Augmentation: {aug_count} images will be augmented "
                f"({self.gallery_augmentations}x each = "
                f"{aug_count * self.gallery_augmentations} extra images)"
            )
        else:
            logger.info("Augmentation disabled, skipping.")
        
        from PIL import Image as PILImage
        
        # Streaming augmentation: process one image at a time,
        # accumulate into small batches, encode, then discard.
        # This avoids loading all augmented images into RAM at once.
        aug_stream_images: List = []
        aug_stream_metas: List = []
        
        def _flush_aug_batch() -> None:
            """Encode and flush current augmentation batch."""
            if not aug_stream_images:
                return
            try:
                results = self.index_builder.encoder.encode_batch(
                    aug_stream_images
                )
                for result, aug_meta in zip(results, aug_stream_metas):
                    if result is not None:
                        aug_emb, _, _ = result
                        embeddings_list.append(aug_emb)
                        metadata_list.append(aug_meta)
                        self.stats["total_augmented_images"] += 1
            except Exception as e:
                logger.error(f"Augmented batch encoding failed: {e}")
            finally:
                for img in aug_stream_images:
                    try:
                        img.close()
                    except Exception:
                        pass
                aug_stream_images.clear()
                aug_stream_metas.clear()
        
        # Count items that need augmentation for progress bar
        items_to_augment = [
            (item, emb) for item, emb in zip(valid_items, embeddings)
            if emb is not None and do_augment
            and item["num_gallery_images"] < self.gallery_augmentation_threshold
        ]
        aug_pbar = tqdm(
            total=len(items_to_augment),
            desc="Augmenting gallery images",
            unit="img"
        ) if items_to_augment else None
        
        for item, embedding in zip(valid_items, embeddings):
            if embedding is None:
                logger.warning(f"Failed to encode: {item['image_path']}")
                continue
            
            embeddings_list.append(embedding)
            
            # Create metadata
            meta = GalleryImageMetadata(
                image_id=item["image_id"],
                image_path=item["image_path"],
                landmark_id=item["landmark_id"],
                landmark_name=item["landmark_name"],
                caption=item["caption"],
                caption_landmark=item["caption_landmark"],
                confidence=item["confidence"],
                mean_conf=item["mean_conf"],
                max_conf=item["max_conf"],
                guide_description=item["guide_description"],
                row_idx=item["row_idx"],
                num_gallery_images=item["num_gallery_images"]
            )
            metadata_list.append(meta)
            
            # Stream augmentation: generate and immediately batch
            should_augment = (
                do_augment
                and item["num_gallery_images"]
                < self.gallery_augmentation_threshold
            )
            
            if should_augment:
                img_path = image_base_dir / item["image_path"]
                try:
                    with PILImage.open(img_path) as orig_img:
                        orig_img = orig_img.convert("RGB")
                        for _ in range(self.gallery_augmentations):
                            try:
                                aug_img = augment_fn(orig_img.copy())
                                aug_stream_images.append(aug_img)
                                aug_stream_metas.append(meta)
                            except Exception as e:
                                logger.debug(f"Augmentation failed: {e}")
                            
                            # Flush when batch is full
                            if len(aug_stream_images) >= self.batch_size:
                                _flush_aug_batch()
                except Exception as e:
                    logger.debug(
                        f"Augmentation skipped for "
                        f"{item['image_path']}: {e}"
                    )
                if aug_pbar is not None:
                    aug_pbar.update(1)
        
        if aug_pbar is not None:
            aug_pbar.close()
        
        # Flush remaining augmented images
        _flush_aug_batch()
        
        if self.stats["total_augmented_images"] > 0:
            logger.info(
                f"Added {self.stats['total_augmented_images']} "
                f"augmented gallery images"
            )
        
        if not embeddings_list:
            raise ValueError("No gallery images were successfully encoded")
        
        # Stack embeddings
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Encoded {len(embeddings)} gallery images")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        
        # Log statistics
        self._log_statistics()
        
        return embeddings, metadata_list, index
    
    def _log_statistics(self) -> None:
        """Log gallery index statistics."""
        logger.info("=" * 60)
        logger.info("GALLERY INDEX STATISTICS")
        if self.stats.get("total_augmented_images", 0) > 0:
            logger.info(
                f"Augmented images added: "
                f"{self.stats['total_augmented_images']}"
            )
        logger.info("=" * 60)
        logger.info(
            f"Total landmarks: {self.stats['total_landmarks']}"
        )
        logger.info(
            f"Total gallery images: {self.stats['total_gallery_images']}"
        )
        logger.info(
            f"  - 1 gallery image: "
            f"{self.stats['landmarks_with_1_image']}"
        )
        logger.info(
            f"  - 2 gallery images: "
            f"{self.stats['landmarks_with_2_images']}"
        )
        logger.info(
            f"  - 3 gallery images: "
            f"{self.stats['landmarks_with_3_images']}"
        )
        logger.info(
            f"  - 4+ gallery images: "
            f"{self.stats['landmarks_with_4plus_images']}"
        )
        logger.info(
            f"Average gallery images per landmark: "
            f"{self.stats['total_gallery_images'] / max(1, self.stats['total_landmarks']):.2f}"
        )
        logger.info("=" * 60)
    
    # Note: _aggregate_scores is now ScoreAggregator.aggregate
    # from src.rag.landmark_retriever
    
    @staticmethod
    def search_and_aggregate(
        query_embedding: np.ndarray,
        gallery_index: faiss.Index,
        gallery_metadata: List[GalleryImageMetadata],
        k: int = 50,
        aggregation_mode: str = "weighted_top2",
        aggregation_alpha: float = 0.7
    ) -> List[Tuple[str, float, List[Tuple[float, GalleryImageMetadata]]]]:
        """
        Search gallery index and aggregate results by landmark_id.
        
        This simulates production retrieval:
        1. FAISS returns top-k gallery images
        2. Group by landmark_id
        3. Aggregate scores per landmark using configurable strategy
        4. Return top landmarks with their gallery images AND scores
        
        Args:
            query_embedding: (D,) query embedding
            gallery_index: FAISS index of gallery images
            gallery_metadata: Metadata for each gallery image
            k: Number of images to retrieve
            aggregation_mode: Score aggregation strategy
            aggregation_alpha: Alpha parameter for weighted_top2
        
        Returns:
            List of (landmark_id, aggregated_score, scores_and_metas) tuples,
            where scores_and_metas is List[Tuple[float, GalleryImageMetadata]],
            sorted by aggregated_score descending
        """
        # Normalize query
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm.reshape(1, -1))
        
        # Search FAISS
        distances, indices = gallery_index.search(
            query_norm.reshape(1, -1), min(k, gallery_index.ntotal)
        )
        
        # Group by landmark_id
        landmark_scores: Dict[
            str, List[Tuple[float, GalleryImageMetadata]]
        ] = {}
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(gallery_metadata):
                continue
            
            meta = gallery_metadata[idx]
            lid = meta.landmark_id
            
            if lid not in landmark_scores:
                landmark_scores[lid] = []
            
            landmark_scores[lid].append((float(dist), meta))
        
        # Aggregate scores per landmark
        landmark_results = []
        for lid, scores_and_metas in landmark_scores.items():
            # Extract scores
            scores = [score for score, _ in scores_and_metas]
            
            # Aggregate using ScoreAggregator from landmark_retriever
            aggregated_score = ScoreAggregator.aggregate(
                scores, mode=aggregation_mode, alpha=aggregation_alpha
            )
            
            # CRITICAL FIX: Keep scores with metadata for downstream selection
            # Return scores_and_metas instead of just gallery_images
            landmark_results.append((lid, aggregated_score, scores_and_metas))
        
        # Sort by aggregated_score descending
        landmark_results.sort(key=lambda x: x[1], reverse=True)
        
        return landmark_results


# ======================
# RETRIEVAL-BASED SAMPLE GENERATOR (NEW)
# ======================
class RetrievalBasedSampleGenerator:
    """
    Generates training samples using natural retrieval.
    
    Key differences from old SampleGenerator:
    - OLD: Manually inject positive candidate
    - NEW: Positive must come from retrieval naturally
    
    - OLD: Query image can be same as positive image
    - NEW: Query image must differ from all candidate images
    
    - OLD: target.name for identification
    - NEW: target_idx for multiple-choice classification
    
    This matches production where:
    1. User uploads query image
    2. System retrieves candidates via FAISS
    3. Reranker selects best match from candidates
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        df: pd.DataFrame,
        gallery_index: faiss.Index,
        gallery_metadata: List[GalleryImageMetadata],
        landmark_splits: List[LandmarkImageSplit],
        index_builder: IndexBuilder
    ):
        self.config = config
        self.df = df
        self.gallery_index = gallery_index
        self.gallery_metadata = gallery_metadata
        self.landmark_splits = landmark_splits
        self.index_builder = index_builder
        
        # Build lookup maps
        self.lid_to_split = {
            split.landmark_id: split for split in landmark_splits
        }
        self.lid_to_row = {
            split.landmark_id: split.row_idx for split in landmark_splits
        }
        
        # Statistics
        self.stats = {
            "total_generated": 0,
            "positive_in_candidates": 0,
            "none_of_the_above": 0,
            "failed_retrieval": 0,
            "query_in_candidates": 0,  # Should be 0!
        }
    
    def _encode_query_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Encode a single query image using shared encoding method."""
        return GalleryIndexBuilder.encode_single_image(
            image_path, self.index_builder.encoder
        )
    
    def _retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        query_landmark_id: str,
        query_image_path: str,
        k: int = 50,
        for_unknown: bool = False,
        for_hard_unknown: bool = False
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieve candidates using FAISS and aggregate by landmark.

        Args:
            query_embedding: Query embedding
            query_landmark_id: Ground truth landmark ID
            query_image_path: Query image path
            k: Number of images to retrieve
            for_unknown: If True, exclude top-k for realistic unknown samples
            for_hard_unknown: If True, keep only candidates with
                retrieval_score >= config.hard_unknown_min_score and
                exclude the correct landmark. Simulates production scenario
                where retrieval is confident but the object is not in DB.

        Returns:
            candidates: List of candidate dicts
            target_idx: Index of correct answer, or -1 if not in candidates
        """
        # Search gallery index with configured aggregation
        results = GalleryIndexBuilder.search_and_aggregate(
            query_embedding=query_embedding,
            gallery_index=self.gallery_index,
            gallery_metadata=self.gallery_metadata,
            k=k,
            aggregation_mode=self.config.score_aggregation_mode,
            aggregation_alpha=self.config.weighted_top2_alpha
        )

        if not results:
            return [], -1

        # For unknown samples: skip top-k to get realistic distractors
        start_rank = 0
        if for_unknown:
            start_rank = self.config.unknown_exclude_topk
            # Ensure we have enough results
            if start_rank >= len(results):
                start_rank = max(0, len(results) - self.config.max_candidates)
        
        # For hard_unknown: filter to only high-score candidates and
        # exclude the correct landmark. This creates the hardest possible
        # unknown scenario: retrieval is very confident, but the object
        # is genuinely not in the database.
        if for_hard_unknown:
            results = [
                (lid, score, imgs)
                for lid, score, imgs in results
                if lid != query_landmark_id
                and score >= self.config.hard_unknown_min_score
            ]
            if not results:
                return [], -1

        # Build candidates from retrieval results
        candidates = []
        target_idx = -1

        for rank, (lid, max_score, gallery_imgs) in enumerate(results):
            # Skip top-k for unknown samples
            if rank < start_rank:
                continue
            # Select one gallery image for this landmark
            # gallery_imgs is List[Tuple[float, GalleryImageMetadata]]
            
            # Sort by individual scores descending
            scores_and_metas_sorted = sorted(
                gallery_imgs,
                key=lambda x: x[0],
                reverse=True
            )
            
            # Filter out query image first (CRITICAL)
            valid_gallery = [
                (score, img_meta)
                for score, img_meta in scores_and_metas_sorted
                if img_meta.image_path != query_image_path
            ]
            
            if not valid_gallery:
                # All gallery images are same as query (shouldn't happen)
                logger.warning(
                    f"Query image {query_image_path} found in all gallery "
                    f"images for landmark {lid}"
                )
                self.stats["query_in_candidates"] += 1
                continue
            
            # Sample from top-k valid images
            topk = min(
                self.config.candidate_image_sample_topk,
                len(valid_gallery)
            )
            candidate_pool = valid_gallery[:topk]
            
            # Randomly select one from the pool (deterministic via seed)
            selected_score, selected_img_meta = random.choice(candidate_pool)
            
            # Determine candidate type based on configured mode
            if self.config.hardness_classification_mode == "score_based":
                # Score-based classification (stable across densities)
                if max_score >= self.config.hard_threshold:
                    cand_type = "hard"
                elif max_score >= self.config.semi_hard_threshold:
                    cand_type = "semi_hard"
                else:
                    cand_type = "easy"
            else:  # rank_based (legacy)
                # Rank-based classification (original behavior)
                if rank < 5:
                    cand_type = "hard"
                elif rank < 15:
                    cand_type = "semi_hard"
                else:
                    cand_type = "easy"
            
            # Create candidate
            candidate = {
                "name": selected_img_meta.landmark_name,
                "landmark_id": lid,
                "image": selected_img_meta.image_path,
                "caption": selected_img_meta.caption,
                "caption_landmark": selected_img_meta.caption_landmark,
                "retrieval_score": float(max_score),  # Aggregated landmark score
                "image_score": float(selected_score),  # Individual image score
                "retrieval_rank": rank,
                "candidate_type": cand_type
            }
            
            candidates.append(candidate)
            
            # Check if this is the target
            if lid == query_landmark_id:
                target_idx = len(candidates) - 1
            
            # Stop if we have enough candidates
            if len(candidates) >= self.config.max_candidates:
                break
        
        return candidates, target_idx
    
    def generate_sample(
        self,
        query_image_path: str,
        query_landmark_id: str,
        image_base_dir: Path,
        force_unknown: bool = False,
        force_hard_unknown: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single training sample.

        Args:
            query_image_path: Path to query image
            query_landmark_id: Ground truth landmark ID
            image_base_dir: Base directory for images
            force_unknown: If True, create none-of-the-above sample
                using outside_topk strategy (low retrieval scores).
            force_hard_unknown: If True, create hard none-of-the-above
                sample using only candidates with retrieval_score >=
                config.hard_unknown_min_score. Simulates production
                scenario where retrieval is confident but object is
                not in the database.

        Returns:
            Sample dict or None if generation failed
        """
        # Encode query image
        full_path = image_base_dir / query_image_path
        query_embedding = self._encode_query_image(full_path)

        if query_embedding is None:
            self.stats["failed_retrieval"] += 1
            return None

        # Retrieve candidates
        candidates, target_idx = self._retrieve_candidates(
            query_embedding=query_embedding,
            query_landmark_id=query_landmark_id,
            query_image_path=query_image_path,
            k=self.config.faiss_k
        )

        if not candidates:
            self.stats["failed_retrieval"] += 1
            return None

        # Decide if this should be a positive or unknown sample
        if force_hard_unknown:
            # Hard unknown: re-retrieve keeping only high-score candidates
            # and excluding the correct landmark.
            candidates, target_idx = self._retrieve_candidates(
                query_embedding=query_embedding,
                query_landmark_id=query_landmark_id,
                query_image_path=query_image_path,
                k=self.config.faiss_k,
                for_hard_unknown=True
            )

            if not candidates:
                # Not enough high-score negatives — skip this sample
                self.stats["failed_retrieval"] += 1
                return None

            # target_idx must be -1 (correct landmark was excluded)
            target_idx = -1

            if len(candidates) > self.config.max_candidates:
                candidates = candidates[:self.config.max_candidates]

            self.stats["none_of_the_above"] += 1

        elif force_unknown:
            # Generate realistic unknown sample
            if self.config.unknown_generation_strategy == "outside_topk":
                # Re-retrieve with for_unknown=True to get realistic
                # distractors (low retrieval scores)
                candidates, target_idx = self._retrieve_candidates(
                    query_embedding=query_embedding,
                    query_landmark_id=query_landmark_id,
                    query_image_path=query_image_path,
                    k=self.config.faiss_k,
                    for_unknown=True
                )

                if not candidates:
                    self.stats["failed_retrieval"] += 1
                    return None

                # target_idx should be -1 (positive not in top-k)
                # But verify and force if needed
                if target_idx != -1:
                    # Positive accidentally in excluded range, remove it
                    candidates.pop(target_idx)
                    target_idx = -1

                # Ensure we have max_candidates
                if len(candidates) > self.config.max_candidates:
                    candidates = candidates[:self.config.max_candidates]

            else:  # manual_removal (legacy)
                # Old behavior: remove target if it's in candidates
                if target_idx != -1:
                    candidates.pop(target_idx)
                    target_idx = -1

                # Ensure we have max_candidates
                if len(candidates) > self.config.max_candidates:
                    candidates = candidates[:self.config.max_candidates]

            self.stats["none_of_the_above"] += 1
        
        elif target_idx == -1:
            # Positive naturally not in candidates (rare)
            # Treat as unknown sample
            if len(candidates) > self.config.max_candidates:
                candidates = candidates[:self.config.max_candidates]
            self.stats["none_of_the_above"] += 1
        
        else:
            # Positive sample
            # Shuffle candidates to randomize target position
            # But keep track of target
            target_candidate = candidates[target_idx]
            target_landmark_id = target_candidate.get("landmark_id")
            random.shuffle(candidates)
            
            # Find new position of target (optimized with landmark_id)
            target_idx = -1
            for idx, cand in enumerate(candidates):
                if cand.get("landmark_id") == target_landmark_id:
                    target_idx = idx
                    break
            
            # Ensure we have exactly max_candidates
            if len(candidates) > self.config.max_candidates:
                # If target would be removed, keep it
                if target_idx >= self.config.max_candidates:
                    # Swap target with last kept candidate
                    candidates[self.config.max_candidates - 1] = (
                        target_candidate
                    )
                    target_idx = self.config.max_candidates - 1
                
                candidates = candidates[:self.config.max_candidates]
            
            self.stats["positive_in_candidates"] += 1
        
        # Get landmark metadata
        split = self.lid_to_split.get(query_landmark_id)
        if not split:
            return None
        
        row = self.df.iloc[split.row_idx]
        
        # Build sample
        sample = {
            "query_image": query_image_path,
            "candidates": candidates,
            "target_idx": target_idx,
            "meta": {
                "landmark_id": query_landmark_id,
                "landmark_name": split.landmark_name,
                "num_candidates": len(candidates),
                "positive_in_candidates": (target_idx != -1),
                "num_images": split.total_images,
                "confidence": split.confidence,
                "mean_conf": split.mean_conf,
                "max_conf": split.max_conf,
                "guide_description": row.get("guide_description", ""),
                "wikidata_id": row.get("wikidata_id", ""),
                "coordinates": row.get("coordinates", {}),
                "country_ru": row.get("country_ru", ""),
                "country_en": row.get("country_en", ""),
                "city_ru": row.get("city_ru", ""),
                "city_en": row.get("city_en", ""),
                "landmark_type": row.get("landmark_type", {}),
                "year_built": row.get("year_built", ""),
                "architectural_style_ru": row.get("architectural_style_ru", []),
                "architectural_style_en": row.get("architectural_style_en", []),
                "heritage_status_ru": row.get("heritage_status_ru", ""),
                "heritage_status_en": row.get("heritage_status_en", ""),
                "wikipedia_url_ru": row.get("wikipedia_url_ru", ""),
                "wikipedia_url_en": row.get("wikipedia_url_en", ""),
                "website": row.get("website", ""),
                "wikidata_description_ru": row.get("wikidata_description_ru", ""),
                "wikidata_description_en": row.get("wikidata_description_en", "")
            }
        }
        
        self.stats["total_generated"] += 1
        return sample
    
    def generate_samples_from_splits(
        self,
        split_type: str,
        image_base_dir: Path,
        unknown_ratio: float = 0.3,
        hard_unknown_ratio: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Generate samples for a specific split (train/val/test).

        Args:
            split_type: "train", "val", or "test"
            image_base_dir: Base directory for images
            unknown_ratio: Fraction of samples that should be
                none-of-the-above (low retrieval score distractors).
            hard_unknown_ratio: Fraction of samples that should be
                hard none-of-the-above (high retrieval score distractors,
                score >= config.hard_unknown_min_score). These teach the
                model to reject even visually similar candidates.
                Evaluated before unknown_ratio — probabilities are
                independent (a sample can only be one type).

        Returns:
            List of samples
        """
        logger.info(f"Generating {split_type} samples...")
        logger.info(
            f"  unknown_ratio={unknown_ratio:.2f}, "
            f"hard_unknown_ratio={hard_unknown_ratio:.2f}"
        )

        samples = []

        for split in tqdm(
            self.landmark_splits, desc=f"Generating {split_type}"
        ):
            # Get query images for this split
            if split_type == "train":
                query_images = split.query_train_images
            elif split_type == "val":
                query_images = split.query_val_images
            elif split_type == "test":
                query_images = split.query_test_images
            else:
                raise ValueError(f"Invalid split_type: {split_type}")

            if not query_images:
                continue

            for query_img in query_images:
                # Determine sample type via independent random draws.
                # hard_unknown is checked first; if not triggered,
                # regular unknown is checked.
                rand_val = random.random()
                if rand_val < hard_unknown_ratio:
                    force_hard_unknown = True
                    force_unknown = False
                elif rand_val < hard_unknown_ratio + unknown_ratio:
                    force_hard_unknown = False
                    force_unknown = True
                else:
                    force_hard_unknown = False
                    force_unknown = False

                sample = self.generate_sample(
                    query_image_path=query_img,
                    query_landmark_id=split.landmark_id,
                    image_base_dir=image_base_dir,
                    force_unknown=force_unknown,
                    force_hard_unknown=force_hard_unknown
                )

                if sample:
                    samples.append(sample)

        logger.info(f"Generated {len(samples)} {split_type} samples")
        self._log_statistics()

        return samples
    
    def _log_statistics(self) -> None:
        """Log generation statistics."""
        logger.info("=" * 60)
        logger.info("SAMPLE GENERATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total generated: {self.stats['total_generated']}")
        logger.info(
            f"Positive in candidates: {self.stats['positive_in_candidates']}"
        )
        logger.info(
            f"None-of-the-above: {self.stats['none_of_the_above']}"
        )
        logger.info(
            f"Failed retrieval: {self.stats['failed_retrieval']}"
        )
        logger.info(
            f"Query in candidates (ERROR): {self.stats['query_in_candidates']}"
        )
        logger.info("=" * 60)


# ======================
# FILE WRITER
# ======================
class FileWriter:
    """Handles writing samples to JSON files."""
    
    @staticmethod
    def save_json(path: Path, data: Any) -> None:
        """Save data to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Log appropriate message based on data type
            if isinstance(data, list):
                logger.info(f"Saved {len(data)} samples to {path}")
            else:
                logger.info(f"Saved data to {path}")
        except Exception as e:
            logger.error(f"Failed to save file {path}: {e}")
            raise


# ======================
# CONTRASTIVE SAMPLE GENERATOR (NEW)
# ======================
class ImageLevelContrastiveGenerator:
    """
    Generates contrastive learning samples with proper image-level constraints.
    
    Key requirements:
    - anchor = query image (from query_images)
    - positive = gallery image of same landmark (from gallery_images)
    - negative = gallery image of different landmark
    - Only for landmarks with >= 3 images (supports_contrastive=True)
    - NO query/gallery leakage
    
    This ensures:
    - Anchor != positive (different images)
    - Positive comes from gallery (not query set)
    - Negative comes from gallery (not query set)
    """
    
    def __init__(self, landmark_splits: List[LandmarkImageSplit]):
        self.landmark_splits = landmark_splits
        
        # Build lookup
        self.lid_to_split = {
            split.landmark_id: split for split in landmark_splits
        }
        
        # Statistics
        self.stats = {
            "total_generated": 0,
            "skipped_no_support": 0,
            "skipped_no_gallery": 0,
            "skipped_no_negative": 0,
        }
    
    def generate_from_samples(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate contrastive samples from reranking samples.
        
        Args:
            samples: List of samples from RetrievalBasedSampleGenerator
        
        Returns:
            List of contrastive triplets
        """
        contrastive_samples = []
        
        for sample in samples:
            # Get landmark info
            landmark_id = sample["meta"]["landmark_id"]
            anchor_image = sample["query_image"]
            
            # Get split info
            split = self.lid_to_split.get(landmark_id)
            if not split:
                continue
            
            # Check if landmark supports contrastive learning
            if not split.supports_contrastive:
                self.stats["skipped_no_support"] += 1
                continue
            
            # Get positive from gallery images
            # CRITICAL: positive must be from gallery, not query
            if not split.gallery_images:
                self.stats["skipped_no_gallery"] += 1
                continue
            
            positive_image = random.choice(split.gallery_images)
            
            # Validate: anchor != positive
            if anchor_image == positive_image:
                logger.warning(
                    f"Anchor equals positive for {landmark_id}: {anchor_image}"
                )
                continue
            
            # Get negative from candidates
            # Prefer candidates from different landmarks
            negative_candidates = [
                c for c in sample["candidates"]
                if c.get("landmark_id") != landmark_id
                and c.get("image")
            ]
            
            if not negative_candidates:
                self.stats["skipped_no_negative"] += 1
                continue
            
            # Select negative (prefer hard negatives)
            negative = random.choice(negative_candidates)
            negative_image = negative["image"]
            
            # Create contrastive sample
            contrastive_sample = {
                "anchor": anchor_image,
                "positive": positive_image,
                "negative": negative_image,
                "landmark_id": landmark_id,
                "landmark_name": split.landmark_name,
                "negative_landmark_id": negative.get("landmark_id", ""),
                "negative_landmark_name": negative.get("name", ""),
                "negative_type": negative.get("candidate_type", "unknown"),
                "negative_retrieval_score": negative.get("retrieval_score", 0.0)
            }
            
            contrastive_samples.append(contrastive_sample)
            self.stats["total_generated"] += 1
        
        self._log_statistics()
        return contrastive_samples
    
    def _log_statistics(self) -> None:
        """Log generation statistics."""
        logger.info("=" * 60)
        logger.info("CONTRASTIVE SAMPLE GENERATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total generated: {self.stats['total_generated']}")
        logger.info(
            f"Skipped (no support): {self.stats['skipped_no_support']}"
        )
        logger.info(
            f"Skipped (no gallery): {self.stats['skipped_no_gallery']}"
        )
        logger.info(
            f"Skipped (no negative): {self.stats['skipped_no_negative']}"
        )
        logger.info("=" * 60)


# ======================
# RETRIEVAL EVALUATION
# ======================
@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    
    # Global metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_10: float = 0.0  # Normalized Discounted Cumulative Gain
    positive_retrieval_rate: float = 0.0
    retrieval_ceiling: float = 0.0  # Max possible recall (positive in top-k)
    
    # Sample counts
    total_queries: int = 0
    queries_with_positive: int = 0
    
    # Hardness distribution
    hard_negatives: int = 0
    semi_hard_negatives: int = 0
    easy_negatives: int = 0
    
    # Per-category metrics (by image count)
    metrics_by_image_count: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "map": self.map_score,
            "ndcg@10": self.ndcg_at_10,
            "positive_retrieval_rate": self.positive_retrieval_rate,
            "retrieval_ceiling": self.retrieval_ceiling,
            "total_queries": self.total_queries,
            "queries_with_positive": self.queries_with_positive,
            "hardness_distribution": {
                "hard": self.hard_negatives,
                "semi_hard": self.semi_hard_negatives,
                "easy": self.easy_negatives
            },
            "by_image_count": self.metrics_by_image_count
        }


class RetrievalEvaluator:
    """Evaluates retrieval performance with detailed metrics."""
    
    def __init__(self, landmark_splits: List[LandmarkImageSplit]):
        self.landmark_splits = landmark_splits
        self.lid_to_image_count = {
            split.landmark_id: split.total_images
            for split in landmark_splits
        }
    
    def _get_image_count_category(self, num_images: int) -> str:
        """Get category label for image count."""
        if num_images == 1:
            return "1_image"
        elif num_images == 2:
            return "2_images"
        elif num_images == 3:
            return "3_images"
        else:
            return "4plus_images"
    
    @staticmethod
    def _compute_average_precision(target_idx: int, num_candidates: int) -> float:
        """Compute Average Precision for single query."""
        if target_idx == -1:
            return 0.0
        rank = target_idx + 1
        return 1.0 / rank
    
    @staticmethod
    def _compute_dcg(relevances: List[float], k: int = 10) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def _compute_ndcg(target_idx: int, num_candidates: int, k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        if target_idx == -1:
            return 0.0
        
        relevances = [0.0] * min(num_candidates, k)
        if target_idx < k:
            relevances[target_idx] = 1.0
        
        dcg = RetrievalEvaluator._compute_dcg(relevances, k)
        ideal_relevances = [1.0] + [0.0] * (k - 1)
        idcg = RetrievalEvaluator._compute_dcg(ideal_relevances, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_samples(
        self,
        samples: List[Dict[str, Any]],
        split_name: str = "unknown"
    ) -> RetrievalMetrics:
        """Evaluate retrieval performance on generated samples."""
        logger.info(f"Evaluating retrieval for {split_name} split...")
        
        metrics = RetrievalMetrics()
        category_stats = {}
        
        for sample in samples:
            query_lid = sample["meta"]["landmark_id"]
            target_idx = sample["target_idx"]
            candidates = sample["candidates"]
            num_images = sample["meta"]["num_images"]
            
            category = self._get_image_count_category(num_images)
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "recall@1": 0,
                    "recall@5": 0,
                    "recall@10": 0,
                    "mrr_sum": 0.0,
                    "map_sum": 0.0,
                    "ndcg_sum": 0.0,
                    "positive_retrieved": 0
                }
            
            metrics.total_queries += 1
            category_stats[category]["total"] += 1
            
            # Count negative hardness distribution
            for cand in candidates:
                cand_type = cand.get("candidate_type", "unknown")
                if cand_type == "hard":
                    metrics.hard_negatives += 1
                elif cand_type == "semi_hard":
                    metrics.semi_hard_negatives += 1
                elif cand_type == "easy":
                    metrics.easy_negatives += 1
            
            if target_idx == -1:
                continue
            
            metrics.queries_with_positive += 1
            category_stats[category]["positive_retrieved"] += 1
            
            rank = target_idx + 1
            
            if rank <= 1:
                metrics.recall_at_1 += 1
                category_stats[category]["recall@1"] += 1
            
            if rank <= 5:
                metrics.recall_at_5 += 1
                category_stats[category]["recall@5"] += 1
            
            if rank <= 10:
                metrics.recall_at_10 += 1
                category_stats[category]["recall@10"] += 1
            
            reciprocal_rank = 1.0 / rank
            metrics.mrr += reciprocal_rank
            category_stats[category]["mrr_sum"] += reciprocal_rank
            
            ap = self._compute_average_precision(target_idx, len(candidates))
            metrics.map_score += ap
            category_stats[category]["map_sum"] += ap
            
            ndcg = self._compute_ndcg(target_idx, len(candidates), k=10)
            metrics.ndcg_at_10 += ndcg
            category_stats[category]["ndcg_sum"] += ndcg
        
        if metrics.total_queries > 0:
            metrics.recall_at_1 /= metrics.total_queries
            metrics.recall_at_5 /= metrics.total_queries
            metrics.recall_at_10 /= metrics.total_queries
            metrics.mrr /= metrics.total_queries
            metrics.map_score /= metrics.total_queries
            metrics.ndcg_at_10 /= metrics.total_queries
            metrics.positive_retrieval_rate = (
                metrics.queries_with_positive / metrics.total_queries
            )
            metrics.retrieval_ceiling = (
                metrics.queries_with_positive / metrics.total_queries
            )
        
        for category, stats in category_stats.items():
            total = stats["total"]
            if total > 0:
                metrics.metrics_by_image_count[category] = {
                    "total_queries": total,
                    "recall@1": stats["recall@1"] / total,
                    "recall@5": stats["recall@5"] / total,
                    "recall@10": stats["recall@10"] / total,
                    "mrr": stats["mrr_sum"] / total,
                    "map": stats["map_sum"] / total,
                    "ndcg@10": stats["ndcg_sum"] / total,
                    "positive_retrieval_rate": (
                        stats["positive_retrieved"] / total
                    )
                }
        
        self._log_metrics(metrics, split_name)
        return metrics
    
    def _log_metrics(self, metrics: RetrievalMetrics, split_name: str) -> None:
        """Log detailed metrics."""
        logger.info("=" * 70)
        logger.info(f"RETRIEVAL EVALUATION: {split_name.upper()}")
        logger.info("=" * 70)
        
        logger.info("Global Metrics:")
        logger.info(f"  Total queries: {metrics.total_queries}")
        logger.info(
            f"  Positive retrieval rate: "
            f"{metrics.positive_retrieval_rate:.3f} "
            f"({metrics.queries_with_positive}/{metrics.total_queries})"
        )
        logger.info(f"  Recall@1:  {metrics.recall_at_1:.3f}")
        logger.info(f"  Recall@5:  {metrics.recall_at_5:.3f}")
        logger.info(f"  Recall@10: {metrics.recall_at_10:.3f}")
        logger.info(f"  MRR:       {metrics.mrr:.3f}")
        logger.info(f"  MAP:       {metrics.map_score:.3f}")
        logger.info(f"  NDCG@10:   {metrics.ndcg_at_10:.3f}")
        logger.info(f"  Retrieval Ceiling: {metrics.retrieval_ceiling:.3f}")
        logger.info("")
        
        total_negatives = (
            metrics.hard_negatives +
            metrics.semi_hard_negatives +
            metrics.easy_negatives
        )
        if total_negatives > 0:
            logger.info("Negative Hardness Distribution:")
            logger.info(
                f"  Hard:      {metrics.hard_negatives} "
                f"({100*metrics.hard_negatives/total_negatives:.1f}%)"
            )
            logger.info(
                f"  Semi-hard: {metrics.semi_hard_negatives} "
                f"({100*metrics.semi_hard_negatives/total_negatives:.1f}%)"
            )
            logger.info(
                f"  Easy:      {metrics.easy_negatives} "
                f"({100*metrics.easy_negatives/total_negatives:.1f}%)"
            )
            logger.info("")
        
        if metrics.metrics_by_image_count:
            logger.info("Metrics by Landmark Image Count:")
            categories = sorted(metrics.metrics_by_image_count.keys())
            
            for category in categories:
                cat_metrics = metrics.metrics_by_image_count[category]
                logger.info(f"  {category}:")
                logger.info(f"    Queries: {cat_metrics['total_queries']}")
                logger.info(
                    f"    Positive rate: "
                    f"{cat_metrics['positive_retrieval_rate']:.3f}"
                )
                logger.info(f"    Recall@1:  {cat_metrics['recall@1']:.3f}")
                logger.info(f"    Recall@5:  {cat_metrics['recall@5']:.3f}")
                logger.info(f"    Recall@10: {cat_metrics['recall@10']:.3f}")
                logger.info(f"    MRR:       {cat_metrics['mrr']:.3f}")
                logger.info(f"    MAP:       {cat_metrics['map']:.3f}")
                logger.info(f"    NDCG@10:   {cat_metrics['ndcg@10']:.3f}")
                logger.info("")
        
        logger.info("=" * 70)
    
    def compare_splits(
        self,
        train_metrics: RetrievalMetrics,
        val_metrics: RetrievalMetrics,
        test_metrics: RetrievalMetrics
    ) -> None:
        """Log comparison of metrics across splits."""
        logger.info("=" * 70)
        logger.info("RETRIEVAL METRICS COMPARISON")
        logger.info("=" * 70)
        
        logger.info(
            f"{'Metric':<25} {'Train':>12} {'Val':>12} {'Test':>12}"
        )
        logger.info("-" * 70)
        
        metrics_to_compare = [
            ("Recall@1", "recall_at_1"),
            ("Recall@5", "recall_at_5"),
            ("Recall@10", "recall_at_10"),
            ("MRR", "mrr"),
            ("MAP", "map_score"),
            ("NDCG@10", "ndcg_at_10"),
            ("Positive Rate", "positive_retrieval_rate")
        ]
        
        for label, attr in metrics_to_compare:
            train_val = getattr(train_metrics, attr)
            val_val = getattr(val_metrics, attr)
            test_val = getattr(test_metrics, attr)
            
            logger.info(
                f"{label:<25} {train_val:>12.3f} "
                f"{val_val:>12.3f} {test_val:>12.3f}"
            )
        
        logger.info("-" * 70)
        logger.info(
            f"{'Total Queries':<25} "
            f"{train_metrics.total_queries:>12} "
            f"{val_metrics.total_queries:>12} "
            f"{test_metrics.total_queries:>12}"
        )
        logger.info("=" * 70)


# ======================
# VALIDATION
# ======================
class DatasetValidator:
    """
    Validates dataset for production-correct closed-set retrieval.
    
    Checks:
    - No gallery-query overlap
    - No query-candidate overlap
    - Target_idx consistency
    - Contrastive triplet validity
    """
    
    @staticmethod
    def validate_splits(splits: List[LandmarkImageSplit]) -> bool:
        """Validate image role splits."""
        logger.info("Validating image role splits...")
        
        errors = []
        for split in splits:
            try:
                split.validate()
            except ValueError as e:
                errors.append(str(e))
        
        if errors:
            logger.error(f"Split validation failed: {len(errors)} errors")
            for err in errors[:10]:  # Show first 10
                logger.error(f"  - {err}")
            return False
        
        logger.info("✓ All splits valid (no gallery-query overlap)")
        return True
    
    @staticmethod
    def validate_samples(samples: List[Dict[str, Any]]) -> bool:
        """Validate generated samples."""
        logger.info(f"Validating {len(samples)} samples...")
        
        errors = []
        query_in_candidates = 0
        
        for i, sample in enumerate(samples):
            query_img = sample.get("query_image")
            candidates = sample.get("candidates", [])
            target_idx = sample.get("target_idx", -1)
            
            # Check query not in candidates
            for cand in candidates:
                if cand.get("image") == query_img:
                    query_in_candidates += 1
                    errors.append(
                        f"Sample {i}: query image in candidates: {query_img}"
                    )
            
            # Check target_idx validity
            if target_idx != -1:
                if target_idx < 0 or target_idx >= len(candidates):
                    errors.append(
                        f"Sample {i}: invalid target_idx {target_idx} "
                        f"for {len(candidates)} candidates"
                    )
        
        if errors:
            logger.error(f"Sample validation failed: {len(errors)} errors")
            for err in errors[:10]:
                logger.error(f"  - {err}")
            return False
        
        if query_in_candidates > 0:
            logger.error(
                f"✗ CRITICAL: {query_in_candidates} samples have "
                "query image in candidates!"
            )
            return False
        
        logger.info("✓ All samples valid (no query-candidate overlap)")
        return True
    
    @staticmethod
    def validate_contrastive(
        contrastive_samples: List[Dict[str, Any]]
    ) -> bool:
        """Validate contrastive samples."""
        logger.info(f"Validating {len(contrastive_samples)} contrastive...")
        
        errors = []
        
        for i, sample in enumerate(contrastive_samples):
            anchor = sample.get("anchor")
            positive = sample.get("positive")
            negative = sample.get("negative")
            
            # Check anchor != positive
            if anchor == positive:
                errors.append(
                    f"Contrastive {i}: anchor equals positive: {anchor}"
                )
            
            # Check anchor != negative
            if anchor == negative:
                errors.append(
                    f"Contrastive {i}: anchor equals negative: {anchor}"
                )
            
            # Check positive != negative
            if positive == negative:
                errors.append(
                    f"Contrastive {i}: positive equals negative: {positive}"
                )
        
        if errors:
            logger.error(
                f"Contrastive validation failed: {len(errors)} errors"
            )
            for err in errors[:10]:
                logger.error(f"  - {err}")
            return False
        
        logger.info("✓ All contrastive samples valid")
        return True
    
    @staticmethod
    def log_dataset_summary(
        splits: List[LandmarkImageSplit],
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        train_contrastive: List[Dict],
        val_contrastive: List[Dict],
        test_contrastive: List[Dict]
    ) -> None:
        """Log comprehensive dataset summary."""
        logger.info("=" * 70)
        logger.info("FINAL DATASET SUMMARY")
        logger.info("=" * 70)
        
        # Splits summary
        total_gallery = sum(len(s.gallery_images) for s in splits)
        total_train_q = sum(len(s.query_train_images) for s in splits)
        total_val_q = sum(len(s.query_val_images) for s in splits)
        total_test_q = sum(len(s.query_test_images) for s in splits)
        
        logger.info(f"Landmarks: {len(splits)}")
        logger.info(f"Gallery images: {total_gallery}")
        logger.info(f"Query images: train={total_train_q}, "
                   f"val={total_val_q}, test={total_test_q}")
        logger.info("")
        
        # Samples summary
        logger.info(f"Reranking samples:")
        logger.info(f"  Train: {len(train_samples)}")
        logger.info(f"  Val: {len(val_samples)}")
        logger.info(f"  Test: {len(test_samples)}")
        logger.info("")
        
        # Contrastive summary
        logger.info(f"Contrastive samples:")
        logger.info(f"  Train: {len(train_contrastive)}")
        logger.info(f"  Val: {len(val_contrastive)}")
        logger.info(f"  Test: {len(test_contrastive)}")
        logger.info("")
        
        # Target distribution
        train_positive = sum(1 for s in train_samples if s["target_idx"] != -1)
        train_unknown = len(train_samples) - train_positive
        
        logger.info(f"Train target distribution:")
        logger.info(f"  Positive: {train_positive} "
                   f"({100*train_positive/len(train_samples):.1f}%)")
        logger.info(f"  Unknown: {train_unknown} "
                   f"({100*train_unknown/len(train_samples):.1f}%)")
        
        logger.info("=" * 70)
    
    @staticmethod
    def evaluate_retrieval_quality(
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        landmark_splits: List[LandmarkImageSplit]
    ) -> Tuple['RetrievalMetrics', 'RetrievalMetrics', 'RetrievalMetrics']:
        """
        Evaluate retrieval quality across all splits.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            landmark_splits: Landmark splits for metadata
        
        Returns:
            Tuple of (train_metrics, val_metrics, test_metrics)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATING RETRIEVAL QUALITY")
        logger.info("=" * 70)
        
        evaluator = RetrievalEvaluator(landmark_splits)
        
        # Evaluate each split
        train_metrics = evaluator.evaluate_samples(train_samples, "train")
        val_metrics = evaluator.evaluate_samples(val_samples, "val")
        test_metrics = evaluator.evaluate_samples(test_samples, "test")
        
        # Compare across splits
        logger.info("")
        evaluator.compare_splits(train_metrics, val_metrics, test_metrics)
        
        return train_metrics, val_metrics, test_metrics


# ======================
# MAIN PIPELINE
# ======================
def main():
    """
    NEW PRODUCTION-CORRECT PIPELINE
    
    Key changes from old pipeline:
    1. Image-level splitting (not landmark-level)
    2. Gallery-only FAISS index
    3. Natural retrieval-based candidates
    4. target_idx format (not target.name)
    5. Proper contrastive constraints
    
    Production behavior simulated:
    User photo → SigLIP → FAISS retrieval → Gallery images
    → Group by landmark → Max similarity → Top-K landmarks
    → Qwen2-VL reranker → Multiple-choice → Prediction
    """
    # Initialize configurations
    dataset_config = DatasetConfig()

    # Resolve model name: use explicit override or per-type default
    _default_models = {
        "siglip": "google/siglip-base-patch16-224",
        "dinov2": "facebook/dinov2-base",
    }
    _model_name = (
        dataset_config.embedder_model
        if dataset_config.embedder_model
        else _default_models.get(dataset_config.embedder_type, "")
    )
    if not _model_name:
        raise ValueError(
            f"Unknown embedder_type '{dataset_config.embedder_type}'. "
            "Supported: 'siglip', 'dinov2'."
        )

    index_config = IndexConfig(
        model_name=_model_name,
        embedder_type=dataset_config.embedder_type,
        batch_size=32,
        max_images_per_landmark=10,
    )
    logger.info(
        f"Embedder: {dataset_config.embedder_type} "
        f"({index_config.model_name})"
    )
    
    # Set random seeds for reproducibility
    random.seed(dataset_config.random_seed)
    np.random.seed(dataset_config.random_seed)
    torch.manual_seed(dataset_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dataset_config.random_seed)
    
    # Set single thread on macOS to prevent segfaults
    if _platform.system() == "Darwin":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logger.info("Set PyTorch to single-threaded mode for macOS")
    
    logger.info("=" * 70)
    logger.info("PRODUCTION-CORRECT DATASET GENERATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Random seed: {dataset_config.random_seed}")
    logger.info("")
    
    try:
        # ============================================================
        # STEP 1: Load and prepare data
        # ============================================================
        data_path = Path(dataset_config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Input file not found: {data_path}")
        
        logger.info("Loading landmark data...")
        with open(dataset_config.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Extract image paths
        def extract_paths(valid_images):
            if isinstance(valid_images, list):
                return [
                    img.get("path") if isinstance(img, dict) else img
                    for img in valid_images
                ]
            return []
        
        df["images"] = df["valid_images"].apply(extract_paths)
        
        # Merge name fields
        if "name_ru" in df.columns:
            df["name"] = df["name_ru"].fillna(df.get("name_en", ""))
        elif "name_en" in df.columns:
            df["name"] = df["name_en"]
        
        # Filter landmarks with images
        df = df[df["images"].map(len) > 0].reset_index(drop=True)
        logger.info(f"Loaded {len(df)} landmarks with images")
        
        # ============================================================
        # STEP 2: Split images into gallery/query roles
        # ============================================================
        logger.info("")
        logger.info("STEP 2: Image role assignment")
        logger.info("-" * 70)
        
        splitter = ImageRoleSplitter(
            config=dataset_config,
            max_gallery_per_landmark=dataset_config.max_gallery_per_landmark
        )
        splits = splitter.split_all_landmarks(df)
        
        # Validate splits
        if not DatasetValidator.validate_splits(splits):
            raise ValueError("Split validation failed!")
        
        # ============================================================
        # STEP 3: Build or load gallery FAISS index
        # ============================================================
        logger.info("")
        logger.info("STEP 3: Gallery index preparation")
        logger.info("-" * 70)
        
        # Validate image directory exists
        image_base_dir = dataset_config.image_base_dir
        if not image_base_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_base_dir}"
            )
        logger.info(f"Using image directory: {image_base_dir}")
        
        # Define training index paths (NOT production index).
        # Include embedder type in filename so siglip and dinov2 indexes
        # can coexist in the same output directory.
        _emb_suffix = dataset_config.embedder_type  # "siglip" or "dinov2"
        training_index_path = (
            dataset_config.output_dir
            / f"training_gallery_index_{_emb_suffix}.faiss"
        )
        training_metadata_path = (
            dataset_config.output_dir
            / f"training_gallery_metadata_{_emb_suffix}.json"
        )
        training_embeddings_path = (
            dataset_config.output_dir
            / f"training_gallery_embeddings_{_emb_suffix}.npy"
        )
        
        # Check if we can reuse existing training index
        _index_exists = training_index_path.exists()
        _meta_exists = training_metadata_path.exists()
        _emb_exists = training_embeddings_path.exists()

        if dataset_config.reuse_training_index:
            if dataset_config.force_rebuild_index:
                logger.info(
                    "Index reuse disabled: force_rebuild_index=True"
                )
            elif not _index_exists:
                logger.info(
                    f"Index reuse skipped: index file not found: "
                    f"{training_index_path}"
                )
            elif not _meta_exists:
                logger.info(
                    f"Index reuse skipped: metadata file not found: "
                    f"{training_metadata_path}"
                )
            elif not _emb_exists:
                logger.info(
                    f"Index reuse skipped: embeddings file not found: "
                    f"{training_embeddings_path}"
                )

        can_reuse = (
            dataset_config.reuse_training_index
            and not dataset_config.force_rebuild_index
            and _index_exists
            and _meta_exists
            and _emb_exists
        )
        
        # Initialize index builder (needed for both build and load)
        index_builder = IndexBuilder(index_config)
        
        if can_reuse:
            # ============================================================
            # STEP 3A: Load existing training index
            # ============================================================
            logger.info("Loading existing training gallery index...")
            logger.info(f"  Index: {training_index_path}")
            logger.info(f"  Metadata: {training_metadata_path}")
            logger.info(f"  Embeddings: {training_embeddings_path}")
            
            try:
                # Load FAISS index
                gallery_index = faiss.read_index(str(training_index_path))
                logger.info(f"Loaded FAISS index: {gallery_index.ntotal} images")
                
                # Load metadata
                with open(training_metadata_path, "r", encoding="utf-8") as f:
                    metadata_dicts = json.load(f)
                gallery_metadata = [
                    GalleryImageMetadata(**m) for m in metadata_dicts
                ]
                logger.info(f"Loaded metadata: {len(gallery_metadata)} entries")
                
                # Load embeddings
                embeddings = np.load(training_embeddings_path)
                logger.info(f"Loaded embeddings: {embeddings.shape}")
                
                # Validate consistency (explicit OR — chained != is unreliable)
                n_faiss = gallery_index.ntotal
                n_meta = len(gallery_metadata)
                n_emb = len(embeddings)
                if n_faiss != n_meta or n_faiss != n_emb:
                    raise ValueError(
                        f"Inconsistent index sizes: "
                        f"FAISS={n_faiss}, "
                        f"metadata={n_meta}, "
                        f"embeddings={n_emb}"
                    )
                
                logger.info("✓ Successfully loaded training index")
                logger.info("  (Use force_rebuild_index=True to rebuild)")
                
            except Exception as e:
                import traceback
                logger.warning(f"Failed to load training index: {e}")
                logger.warning(traceback.format_exc())
                logger.info("Falling back to building new index...")
                can_reuse = False
        
        if not can_reuse:
            # ============================================================
            # STEP 3B: Build new gallery index
            # ============================================================
            logger.info("Building new training gallery index...")
            
            try:
                # Build gallery index with batch encoding + augmentation
                gallery_builder = GalleryIndexBuilder(
                    index_builder,
                    batch_size=dataset_config.encoding_batch_size,
                    gallery_augmentations=(
                        dataset_config.gallery_augmentations
                    ),
                    gallery_augmentation_threshold=(
                        dataset_config.gallery_augmentation_threshold
                    )
                )
                embeddings, gallery_metadata, gallery_index = (
                    gallery_builder.build_from_splits(
                        splits=splits,
                        df=df,
                        image_base_dir=image_base_dir
                    )
                )
                
                logger.info(f"Gallery index built: {gallery_index.ntotal} images")
                
            except Exception as e:
                logger.error(f"Failed to build gallery index: {e}")
                raise
        
        # ============================================================
        # STEP 4: Generate reranking samples
        # ============================================================
        logger.info("")
        logger.info("STEP 4: Generating reranking samples")
        logger.info("-" * 70)
        
        sample_generator = RetrievalBasedSampleGenerator(
            config=dataset_config,
            df=df,
            gallery_index=gallery_index,
            gallery_metadata=gallery_metadata,
            landmark_splits=splits,
            index_builder=index_builder
        )
        
        # Generate for each split.
        # hard_unknown_ratio is passed from config — train/val/test all get
        # hard unknown samples to teach and evaluate the model's ability to
        # reject high-confidence false positives (retrieval_score >= threshold).
        train_samples = sample_generator.generate_samples_from_splits(
            split_type="train",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio
        )

        val_samples = sample_generator.generate_samples_from_splits(
            split_type="val",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio
        )

        test_samples = sample_generator.generate_samples_from_splits(
            split_type="test",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio
        )
        
        # Validate samples
        logger.info("")
        if not DatasetValidator.validate_samples(train_samples):
            raise ValueError("Train sample validation failed!")
        if not DatasetValidator.validate_samples(val_samples):
            raise ValueError("Val sample validation failed!")
        if not DatasetValidator.validate_samples(test_samples):
            raise ValueError("Test sample validation failed!")
        
        # Evaluate retrieval quality
        train_metrics, val_metrics, test_metrics = (
            DatasetValidator.evaluate_retrieval_quality(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                landmark_splits=splits
            )
        )
        
        # ============================================================
        # STEP 5: Generate contrastive samples
        # ============================================================
        logger.info("")
        logger.info("STEP 5: Generating contrastive samples")
        logger.info("-" * 70)
        
        contrastive_gen = ImageLevelContrastiveGenerator(splits)
        
        train_contrastive = contrastive_gen.generate_from_samples(train_samples)
        val_contrastive = contrastive_gen.generate_from_samples(val_samples)
        test_contrastive = contrastive_gen.generate_from_samples(test_samples)
        
        # Validate contrastive
        logger.info("")
        if not DatasetValidator.validate_contrastive(train_contrastive):
            raise ValueError("Train contrastive validation failed!")
        if not DatasetValidator.validate_contrastive(val_contrastive):
            raise ValueError("Val contrastive validation failed!")
        if not DatasetValidator.validate_contrastive(test_contrastive):
            raise ValueError("Test contrastive validation failed!")
        
        # ============================================================
        # STEP 6: Save datasets
        # ============================================================
        logger.info("")
        logger.info("STEP 6: Saving datasets")
        logger.info("-" * 70)
        
        FileWriter.save_json(
            dataset_config.output_dir / "train.json", train_samples
        )
        FileWriter.save_json(
            dataset_config.output_dir / "val.json", val_samples
        )
        FileWriter.save_json(
            dataset_config.output_dir / "test.json", test_samples
        )
        
        FileWriter.save_json(
            dataset_config.output_dir / "train_contrastive.json",
            train_contrastive
        )
        FileWriter.save_json(
            dataset_config.output_dir / "val_contrastive.json",
            val_contrastive
        )
        FileWriter.save_json(
            dataset_config.output_dir / "test_contrastive.json",
            test_contrastive
        )
        
        # ============================================================
        # STEP 6.5: Save training gallery index for reuse
        # ============================================================
        if not can_reuse:  # Only save if we just built it
            logger.info("")
            logger.info("STEP 6.5: Saving training gallery index")
            logger.info("-" * 70)
            logger.info("NOTE: This is a TRAINING index (gallery images only)")
            logger.info("      For production, use LandmarkRetriever.build_index_from_landmarks()")
            
            try:
                # Save FAISS index
                logger.info(f"Saving FAISS index to {training_index_path}")
                faiss.write_index(gallery_index, str(training_index_path))
                
                # Save gallery metadata
                logger.info(f"Saving metadata to {training_metadata_path}")
                metadata_dicts = [meta.to_dict() for meta in gallery_metadata]
                FileWriter.save_json(training_metadata_path, metadata_dicts)
                
                # Save embeddings (for analysis and consistency checks)
                logger.info(f"Saving embeddings to {training_embeddings_path}")
                np.save(training_embeddings_path, embeddings)
                
                logger.info(f"✓ Training index saved: {gallery_index.ntotal} images")
                logger.info(f"  Index size: {training_index_path.stat().st_size / 1024 / 1024:.2f} MB")
                logger.info(f"  Embeddings size: {training_embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")
                logger.info("")
                logger.info("To reuse this index in future runs:")
                logger.info("  - Set reuse_training_index=True (default)")
                logger.info("  - Set force_rebuild_index=False (default)")
                logger.info("To force rebuild:")
                logger.info("  - Set force_rebuild_index=True")
                
            except Exception as e:
                logger.warning(f"Failed to save training index: {e}")
                logger.warning("Continuing without saving index...")
        
        # ============================================================
        # STEP 7: Save evaluation metrics
        # ============================================================
        logger.info("")
        logger.info("STEP 7: Saving evaluation metrics")
        logger.info("-" * 70)
        
        # Save metrics to JSON
        metrics_data = {
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "test": test_metrics.to_dict()
        }
        
        FileWriter.save_json(
            dataset_config.output_dir / "retrieval_metrics.json",
            metrics_data
        )
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        logger.info("")
        DatasetValidator.log_dataset_summary(
            splits=splits,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_contrastive=train_contrastive,
            val_contrastive=val_contrastive,
            test_contrastive=test_contrastive
        )
        
        logger.info("")
        logger.info("✓ Dataset generation completed successfully!")
        logger.info(f"✓ Output directory: {dataset_config.output_dir}")
        logger.info("✓ All validation checks passed")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up resources to prevent semaphore leak warnings
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    main()

    # index_config = IndexConfig( 
    #     embedder_type="dinov2",
    #     batch_size=32,
    #     max_images_per_landmark=10)
    # build_index_from_landmarks(
    #     landmarks_json_path="data/processed/landmarks_with_guide_descriptions_filtred.json",
    #     image_base_dir="images",
    #     output_dir="data/index/dinov2",
    #     index_config=index_config
    # )
