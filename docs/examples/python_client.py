#!/usr/bin/env python3
"""
Example Python client for AI Tour Guide API.

Usage:
    python python_client.py path/to/image.jpg
"""

import sys
import requests
from pathlib import Path
from typing import Optional, Dict


class AITourGuideClient:
    """Client for AI Tour Guide API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def predict(
        self,
        image_path: str,
        use_internet_search: bool = True,
        timeout: int = 120
    ) -> Dict:
        """
        Predict landmark from image.
        
        Args:
            image_path: Path to image file
            use_internet_search: Enable internet search fallback
            timeout: Request timeout in seconds
        
        Returns:
            Prediction result dictionary
        
        Raises:
            FileNotFoundError: If image file not found
            requests.HTTPError: If API request fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        url = f"{self.base_url}/v1/predict"
        
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/jpeg")}
            data = {"use_internet_search": str(use_internet_search).lower()}
            
            response = self.session.post(
                url,
                files=files,
                data=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            return response.json()
    
    def health_check(self) -> Dict:
        """
        Check API health.
        
        Returns:
            Health status dictionary
        """
        url = f"{self.base_url}/v1/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict:
        """
        Get API information.
        
        Returns:
            API info dictionary
        """
        url = f"{self.base_url}/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def main():
    """Main function for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python python_client.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize client
    client = AITourGuideClient()
    
    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"✓ API Status: {health['status']}")
    except requests.RequestException as e:
        print(f"✗ API not available: {e}")
        sys.exit(1)
    
    # Predict landmark
    print(f"\nAnalyzing image: {image_path}")
    try:
        result = client.predict(image_path)
        
        print("\n" + "="*60)
        print(f"Landmark: {result['name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Source: {result['source']}")
        print("\nDescription:")
        print(result['description'])
        print("\nTiming:")
        for key, value in result['timing'].items():
            print(f"  {key}: {value:.3f}s")
        print("="*60)
        
    except requests.HTTPError as e:
        print(f"✗ Prediction failed: {e}")
        if e.response is not None:
            print(f"  Error: {e.response.json().get('detail', 'Unknown error')}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
