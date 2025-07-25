import requests
import json
import base64
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_images/sample_waste.jpg"  # Add your test image

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/v1/detection/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_detection(image_path: str):
    """Test detection endpoint"""
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        print("Please add a test image or update the path")
        return
    
    print(f"Testing detection with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'confidence_threshold': 0.3,
            'iou_threshold': 0.45,
            'return_image': True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detection/detect",
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Detection count: {result['detection_count']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        
        for i, detection in enumerate(result['detections']):
            print(f"Detection {i+1}:")
            print(f"  Class: {detection['class_name']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  Bbox: {detection['bbox']}")
        
        # Save annotated image if available
        if result.get('annotated_image'):
            print("Saving annotated image...")
            # Decode base64 image
            image_data = result['annotated_image'].split(',')[1]
            with open('test_result_annotated.jpg', 'wb') as f:
                f.write(base64.b64decode(image_data))
            print("Annotated image saved as 'test_result_annotated.jpg'")
    else:
        print(f"Error: {response.text}")
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("=== API Testing Script ===")
    print(f"Base URL: {API_BASE_URL}")
    print("=" * 50)
    
    # Test endpoints
    test_health()
    test_model_info()
    test_detection(TEST_IMAGE_PATH)
    
    print("Testing completed!")

if __name__ == "__main__":
    main()