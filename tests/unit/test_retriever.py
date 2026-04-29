# tests/unit/test_retriever.py
"""
Unit tests for RAG retriever module.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestRAGRetriever:
    """Tests for RAGRetriever class."""
    
    @pytest.fixture
    def mock_retriever_dependencies(self):
        """Mock dependencies for RAGRetriever."""
        with patch("rag.retriever.faiss") as mock_faiss, \
             patch("rag.retriever.CLIPModel") as mock_clip_model, \
             patch("rag.retriever.CLIPProcessor") as mock_clip_processor, \
             patch("builtins.open", create=True) as mock_open, \
             patch("pickle.load") as mock_pickle:
            
            # Mock FAISS index
            mock_index = MagicMock()
            mock_index.d = 768
            mock_faiss.read_index.return_value = mock_index
            
            # Mock metadata
            mock_pickle.return_value = {
                "lid_list": ["1", "2", "3"],
                "model_name": "openai/clip-vit-large-patch14"
            }
            
            yield {
                "faiss": mock_faiss,
                "clip_model": mock_clip_model,
                "clip_processor": mock_clip_processor,
                "index": mock_index,
            }
    
    def test_encode_image(self, mock_retriever_dependencies, sample_image):
        """Test image encoding."""
        from rag.retriever import RAGRetriever
        
        with patch.object(RAGRetriever, '__init__', lambda x, **kwargs: None):
            retriever = RAGRetriever.__new__(RAGRetriever)
            retriever.clip_model = Mock()
            retriever.clip_processor = Mock()
            retriever.device = "cpu"
            
            # Mock processor output
            retriever.clip_processor.return_value = Mock()
            
            # Mock model output
            mock_features = Mock()
            mock_features.cpu.return_value.numpy.return_value.astype.return_value = \
                np.random.rand(1, 768).astype("float32")
            retriever.clip_model.get_image_features.return_value = mock_features
            
            embedding = retriever.encode_image(sample_image)
            
            assert embedding.shape == (1, 768)
            assert embedding.dtype == np.float32
    
    def test_format_context(self, sample_retrieved_results):
        """Test context formatting."""
        from rag.retriever import RAGRetriever
        
        with patch.object(RAGRetriever, '__init__', lambda x, **kwargs: None):
            retriever = RAGRetriever.__new__(RAGRetriever)
            
            context = retriever.format_context(sample_retrieved_results)
            
            assert "Эйфелева башня" in context
            assert "Колизей" in context
            assert "Тадж-Махал" in context
            assert context.startswith("1.")
    
    def test_format_context_with_max_length(self, sample_retrieved_results):
        """Test context formatting with max description length."""
        from rag.retriever import RAGRetriever
        
        with patch.object(RAGRetriever, '__init__', lambda x, **kwargs: None):
            retriever = RAGRetriever.__new__(RAGRetriever)
            
            context = retriever.format_context(sample_retrieved_results, max_desc_len=10)
            
            # Check that descriptions are truncated
            assert "..." in context
    
    def test_get_facts(self, sample_facts_db):
        """Test getting facts by landmark ID."""
        from rag.retriever import RAGRetriever
        
        with patch.object(RAGRetriever, '__init__', lambda x, **kwargs: None):
            retriever = RAGRetriever.__new__(RAGRetriever)
            retriever.facts_db = sample_facts_db
            
            facts = retriever.get_facts("1")
            
            assert facts["name_ru"] == "Эйфелева башня"
            assert facts["name_en"] == "Eiffel Tower"
    
    def test_get_facts_missing_id(self, sample_facts_db):
        """Test getting facts for non-existent landmark ID."""
        from rag.retriever import RAGRetriever
        
        with patch.object(RAGRetriever, '__init__', lambda x, **kwargs: None):
            retriever = RAGRetriever.__new__(RAGRetriever)
            retriever.facts_db = sample_facts_db
            
            facts = retriever.get_facts("999")
            
            assert facts == {}


@pytest.mark.unit
class TestEmbeddingNormalization:
    """Tests for embedding normalization."""
    
    def test_embedding_normalization(self):
        """Test that embeddings are L2 normalized."""
        embedding = np.random.rand(1, 768).astype("float32")
        
        # Normalize
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        normalized = embedding / norm
        
        # Check L2 norm is 1
        result_norm = np.linalg.norm(normalized, axis=-1)
        assert np.allclose(result_norm, 1.0)
