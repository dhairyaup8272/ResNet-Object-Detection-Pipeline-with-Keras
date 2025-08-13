# ResNet-Object-Detection-Pipeline-with-Keras

This project contains an end-to-end object detection pipeline built with TensorFlow 2.x. It goes beyond simple bounding box prediction by performing deep analysis on each detected Region of Interest (ROI), extracting rich metadata, and storing the results in a cloud backend. The architecture is designed to be modular and extensible, making it an excellent foundation for production-level computer vision systems.

The script demonstrates a modern two-stage detection process, including a ResNet-like backbone for feature extraction and a dedicated ROI head with parallel branches for classification and bounding box regression. Its standout feature is the detailed analysis performed on each final detection, capturing insights on image quality, color composition, and spatial properties.

## Features
**Modular Two-Stage Architecture**: Follows the proven design of a feature backbone and a detection head, allowing components to be upgraded independently.

**ResNet-like Backbone**: Uses a powerful residual network for robust feature extraction, capable of handling variations in input image sizes.

**Advanced ROI Head**: Employs parallel branches to simultaneously classify the object's identity ("what is it?") and regress its bounding box coordinates ("where is it?").

**Metadata Extraction**: Each detected object is analyzed for:
  
  Image Quality: Assesses sharpness, contrast, brightness, and noise.
  
  Color Composition: Extracts dominant colors using K-Means clustering.
  
  Spatial Analysis: Calculates the object's size relative to the entire image.
  
  Pluggable Cloud Storage Backend: Features a storage interface (StorageInterface) with a ready-to-use AWS S3 implementation (S3Storage) for saving detected ROIs and their JSON metadata.
  
  Non-Maximum Suppression (NMS): Includes a from-scratch NMS implementation to clean up overlapping detections and produce a final, precise output.
  
  Extensible Design: Built with modern Python features like dataclasses and abstract base classes, making it easy to extend with new models, analysis functions, or storage backends (e.g., Google Cloud Storage).

## Architecture Overview
The pipeline processes an image through a logical sequence of steps to produce enriched, storable detections

```Input Image
     |
     v
ResNet Backbone
(Feature Extraction)
     |
     v
Feature Map
     |
     v
Region Proposal Generation
(Simulated)
     |
     v
ROI Pooling
(Simulated)
     |
     v
ROI Head
(Classification + BBox Regression)
     |
     v
Non-Maximum Suppression (NMS)
(Filtering Duplicates)
     |
     v
Rich Metadata Analysis
(Quality, Color, Size)
     |
     v
Cloud Storage
(e.g., AWS S3)
```

## Code Components
EnhancedObjectDetectionPipeline: The main orchestrator class that manages the entire workflow.

BackboneNetwork: Contains the logic for the ResNet-like feature extractor. This can be swapped out with other backbones like EfficientNet or MobileNet.

ROIHead: A Keras model with two output branches for classification and regression, including a compiled loss function to train both tasks jointly.

ImageAnalyzer: A static utility class that holds all the "rich metadata" extraction logic. New analysis functions (e.g., blur detection, lighting direction) can be added here.

StorageInterface / S3Storage: An example of the Strategy design pattern. To add a new backend (like GCP or Azure Blob Storage), simply create a new class that inherits from StorageInterface and implement the save_roi method.

## Extensibility

Some ways to extend this model include:

  Implement a GCP Storage Backend: Create a GCPStorage class that implements the StorageInterface and complete the conceptual replicate_to_gcp_storage function for cross-cloud workflows.

  Add a Real Proposal Network: Replace the simulated generate_roi_proposals function with a Region Proposal Network (RPN) to create a full Faster R-CNN implementation.

  Train with Real Data: Adapt the pipeline to load a real dataset (your own, ImageNet, COCO) and implement proper training for both the backbone and the ROI head.

  Add More Analyzers: Enhance the ImageAnalyzer with more sophisticated functions, such as estimating the lighting conditions, detecting text (OCR), or identifying image composition styles.
