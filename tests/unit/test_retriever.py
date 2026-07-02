# tests/unit/test_retriever.py
"""Юнит-тесты LandmarkRetriever и вспомогательной логики."""

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestEmbeddingNormalization:
    """Проверяем L2-нормализацию эмбеддингов."""

    def test_нормализованный_вектор_имеет_норму_1(self):
        """После L2-нормализации норма вектора должна быть равна 1."""
        vec = np.random.rand(1, 768).astype("float32")
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        normalized = vec / norm

        result_norm = np.linalg.norm(normalized, axis=-1)
        assert np.allclose(result_norm, 1.0, atol=1e-6)

    def test_нулевой_вектор_не_нормализуется(self):
        """Нулевой вектор нельзя нормализовать — проверяем что не падает."""
        vec = np.zeros((1, 768), dtype="float32")
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        # norm == 0, деление даст nan/inf — это ожидаемое поведение
        assert norm[0, 0] == 0.0


class TestLandmarkRetrieverUnit:
    """Юнит-тесты LandmarkRetriever без загрузки реальных моделей."""

    @pytest.fixture
    def mock_retriever(self):
        """
        Создаём экземпляр LandmarkRetriever с замоканными зависимостями
        (SigLIP-модель, FAISS-индекс, метаданные).
        """
        from src.rag.landmark_retriever import (
            GalleryImageMetadata,
            LandmarkRetriever,
        )

        mock_index = MagicMock()
        mock_index.d = 768
        mock_index_builder = MagicMock()

        # Создаём минимальные метаданные нужного типа
        meta1 = GalleryImageMetadata(
            image_id=0,
            image_path="eiffel.jpg",
            landmark_id="1",
            landmark_name="Eiffel Tower",
            caption_landmark="",
            caption="",
        )
        meta2 = GalleryImageMetadata(
            image_id=1,
            image_path="colosseum.jpg",
            landmark_id="2",
            landmark_name="Colosseum",
            caption_landmark="",
            caption="",
        )

        retriever = LandmarkRetriever.__new__(LandmarkRetriever)
        retriever.gallery_index = mock_index
        retriever.gallery_metadata = [meta1, meta2]
        retriever.index_builder = mock_index_builder
        retriever.aggregation_mode = "weighted_top2"
        retriever.aggregation_alpha = 0.7
        return retriever

    def test_gallery_metadata_не_пустой(self, mock_retriever):
        """Метаданные галереи должны содержать хотя бы один объект."""
        assert len(mock_retriever.gallery_metadata) > 0

    def test_encode_query_вызывает_encoder(self, mock_retriever, sample_image):
        """_encode_query должен вызывать index_builder.encoder.encode_batch."""
        vec = np.random.rand(1, 768).astype("float32")
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

        # encode_batch возвращает список кортежей (embedding, score, flag)
        mock_retriever.index_builder.encoder.encode_batch = MagicMock(
            return_value=[(vec, 0.9, True)]
        )

        result = mock_retriever._encode_query(sample_image)

        mock_retriever.index_builder.encoder.encode_batch.assert_called_once()
        assert result is not None
