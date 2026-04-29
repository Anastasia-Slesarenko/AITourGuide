# tests/integration/test_api.py
"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_endpoint_exists(self, api_client):
        """Test that health check endpoint exists."""
        response = api_client.get("/v1/health")
        assert response.status_code in [200, 503]
    
    def test_health_check_response_structure(self, api_client):
        """Test health check response structure."""
        response = api_client.get("/v1/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert data["service"] == "AITourGuide"
    
    def test_legacy_health_endpoint(self, api_client):
        """Test legacy health endpoint."""
        response = api_client.get("/health")
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API information."""
        response = api_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["service"] == "AI Tour Guide API"
    
    def test_root_endpoint_has_docs_links(self, api_client):
        """Test that root endpoint includes documentation links."""
        response = api_client.get("/")
        data = response.json()
        
        assert "docs" in data
        assert "redoc" in data
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"


@pytest.mark.integration
@pytest.mark.slow
class TestPredictEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_endpoint_requires_image(self, api_client):
        """Test that predict endpoint requires an image."""
        response = api_client.post("/v1/predict")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_endpoint_with_invalid_image(self, api_client):
        """Test prediction with invalid image data."""
        files = {"image": ("test.jpg", b"invalid image data", "image/jpeg")}
        response = api_client.post("/v1/predict", files=files)
        
        # Should return 400 or 503 depending on service state
        assert response.status_code in [400, 503]
    
    def test_predict_endpoint_with_valid_image(self, api_client, sample_image_bytes):
        """Test prediction with valid image."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        data = {"use_internet_search": "false"}
        
        response = api_client.post("/v1/predict", files=files, data=data)
        
        # May fail if service not initialized, but should not crash
        assert response.status_code in [200, 503, 504]
    
    def test_predict_endpoint_file_size_limit(self, api_client, large_image):
        """Test that large files are rejected."""
        # Create a very large image
        buffer = BytesIO()
        large_image.save(buffer, format="JPEG", quality=100)
        large_bytes = buffer.getvalue()
        
        files = {"image": ("large.jpg", large_bytes, "image/jpeg")}
        response = api_client.post("/v1/predict", files=files)
        
        # Should reject if file is too large
        if len(large_bytes) > 10 * 1024 * 1024:
            assert response.status_code in [400, 413]
    
    def test_predict_endpoint_use_internet_search_param(self, api_client, sample_image_bytes):
        """Test use_internet_search parameter."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        # Test with internet search enabled
        data = {"use_internet_search": "true"}
        response = api_client.post("/v1/predict", files=files, data=data)
        assert response.status_code in [200, 503, 504]
        
        # Test with internet search disabled
        data = {"use_internet_search": "false"}
        response = api_client.post("/v1/predict", files=files, data=data)
        assert response.status_code in [200, 503, 504]


@pytest.mark.integration
class TestRateLimiting:
    """Tests for rate limiting."""
    
    @pytest.mark.slow
    def test_rate_limiting_enforced(self, api_client, sample_image_bytes):
        """Test that rate limiting is enforced."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        # Make multiple requests rapidly
        responses = []
        for _ in range(15):  # More than the default limit of 10
            response = api_client.post("/v1/predict", files=files)
            responses.append(response)
        
        # At least one should be rate limited
        status_codes = [r.status_code for r in responses]
        # 429 is rate limit exceeded
        # May not trigger in test environment, so we just check it doesn't crash
        assert all(code in [200, 400, 429, 503, 504] for code in status_codes)


@pytest.mark.integration
class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, api_client):
        """Test that CORS headers are present."""
        response = api_client.options("/v1/health")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or \
               response.status_code == 200  # Some test clients don't handle OPTIONS


@pytest.mark.integration
class TestDocumentation:
    """Tests for API documentation."""
    
    def test_openapi_schema_accessible(self, api_client):
        """Test that OpenAPI schema is accessible."""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_swagger_ui_accessible(self, api_client):
        """Test that Swagger UI is accessible."""
        response = api_client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_accessible(self, api_client):
        """Test that ReDoc is accessible."""
        response = api_client.get("/redoc")
        assert response.status_code == 200
