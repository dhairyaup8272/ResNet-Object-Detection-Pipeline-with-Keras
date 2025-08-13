import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import uuid
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cv2
from sklearn.cluster import KMeans

# --- AWS S3 Imports ---
import boto3
from botocore.exceptions import ClientError

# --- GCP Storage Imports, if you chose a different cloud ---
# from google.cloud import storage # pip install google-cloud-storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration class for detection pipeline parameters."""
    roi_fixed_size: Tuple[int, int, int] = (64, 64, 3)
    num_classes: int = 5
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and metadata"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    """These variables store metadata of note. In this case, we're interested in specifics on incoming images. Further extrapolation could be made in warehousing based on captured metdata"""
    class_name: str = ""
    dominant_colors: List[Tuple[int, int, int]] = None
    photo_quality_score: float = 0.0
    size_percentage: float = 0.0
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height

class StorageInterface(ABC):
    """Abstract interface for storage backends."""
    
    @abstractmethod
    def save_roi(self, roi_data: np.ndarray, metadata: Dict) -> str:
        pass


class S3Storage(StorageInterface):
    """
    Implements storage to an AWS S3 bucket.
    Requires AWS credentials configured (e.g., via environment variables, ~/.aws/credentials).
    """
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        # Verify bucket existence or create it (optional, but good for setup)
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"S3 bucket '{self.bucket_name}' does not exist. Attempting to create it.")
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"S3 bucket '{self.bucket_name}' created successfully.")
                except ClientError as ce:
                    logger.error(f"Failed to create S3 bucket '{self.bucket_name}': {ce}")
                    raise
            else:
                logger.error(f"Error accessing S3 bucket '{self.bucket_name}': {e}")
                raise

    def save_roi(self, roi_data: np.ndarray, metadata: Dict) -> str:
        """Save ROI data and metadata to S3. Depending on sensitivity, it may be wise to configure specific ACLs, Bucket Policies, or Service Principals to limit access to the bucket if they aren't already defined by SCP or another hierarchical definition."""
        class_name = metadata.get('class_name', 'unknown').replace(" ", "_") # Sanitize for S3 key
        unique_id = f"{class_name}_{uuid.uuid4().hex}"
        
        # S3 key for the image data (e.g., class_name/unique_id.npy)
        image_key = f"{class_name}/{unique_id}.npy"
        # S3 key for the metadata (e.g., class_name/unique_id_metadata.json)
        metadata_key = f"{class_name}/{unique_id}_metadata.json"

        try:
            # Save numpy array to a BytesIO object and upload
            with io.BytesIO() as buffer:
                np.save(buffer, roi_data)
                buffer.seek(0) # Rewind to the beginning of the buffer
                self.s3_client.upload_fileobj(buffer, self.bucket_name, image_key)

            # Upload metadata as a JSON file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            s3_uri = f"s3://{self.bucket_name}/{class_name}/{unique_id}"
            logger.info(f"Saved ROI {unique_id} to S3: {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Failed to save ROI {unique_id} to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during S3 upload: {e}")
            raise

# --- Conceptual GCP Storage Replication Function ---
# def replicate_to_gcp_storage(aws_s3_uri: str, gcp_bucket_name: str):
#     """
#     Conceptually replicates an object from an AWS S3 URI to Google Cloud Storage.
#     This function would typically be triggered by an event (e.g., S3 event notification
#     to a Lambda, which then calls this function). It's useful for many cross-cloud use cases, especially B2B SAAS applications.
#     Also of note is data egress costs, which should be communicated to customers or business partners in your organization.
#     
#     Args:
#         aws_s3_uri: The S3 URI of the object to replicate (e.g., "s3://my-bucket/path/to/object.npy").
#         gcp_bucket_name: The name of the GCP bucket to replicate to.
#     """
#     logger.info(f"Attempting to replicate {aws_s3_uri} to gs://{gcp_bucket_name}")
#     
#     # Parse S3 URI
#     s3_parts = aws_s3_uri.replace("s3://", "").split('/', 1)
#     s3_bucket = s3_parts[0]
#     s3_key = s3_parts[1] if len(s3_parts) > 1 else ""
#     
#     # Initialize S3 client to download (if needed, or directly copy if cross-cloud copy is supported)
#     s3_client = boto3.client('s3')
#     
#     # Initialize GCP Storage client
#     # storage_client = storage.Client()
#     # gcp_bucket = storage_client.bucket(gcp_bucket_name)
#     
#     # --- Conceptual Replication Logic ---
#     try:
#         # 1. Download object from S3 (or stream directly if possible)
#         # s3_object = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
#         # s3_data = s3_object['Body'].read()
#         
#         # 2. Upload object to GCP Storage
#         # gcp_blob = gcp_bucket.blob(s3_key) # Use same key structure
#         # gcp_blob.upload_from_string(s3_data)
#         
#         logger.info(f"Successfully replicated {s3_key} from S3 to GCP bucket {gcp_bucket_name}")
#         
#         # Optionally, replicate associated metadata file as well
#         # metadata_key = s3_key.replace(".npy", "_metadata.json")
#         # s3_metadata_object = s3_client.get_object(Bucket=s3_bucket, Key=metadata_key)
#         # s3_metadata = s3_metadata_object['Body'].read()
#         # gcp_metadata_blob = gcp_bucket.blob(metadata_key)
#         # gcp_metadata_blob.upload_from_string(s3_metadata, content_type='application/json')
#         # logger.info(f"Successfully replicated metadata for {s3_key} to GCP.")
#         
#     except ClientError as e:
#         logger.error(f"AWS S3 error during GCP replication: {e}")
#     # except Exception as e: # Catch GCP client errors
#     #     logger.error(f"GCP storage error during replication: {e}")
#     
#     logger.warning("GCP replication function is conceptual and requires actual implementation.")


class BackboneNetwork:
    """Improved backbone network with modern architecture."""
    
    @staticmethod
    def create_resnet_backbone(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create a ResNet-like backbone for feature extraction. ResNet was designed specifically for computer vision tasks. Residual connections help with gradient flow and and training stabilization in deep networks, which is especially important for object detection tasks. ResNet is known as the backbone of AlphaGo and other well known transformers. Similar constructs have been mapped to insect brains in biology. It's interesting to think about what future architectures may look like when comparing model constructs to pre-existing biological constructs."""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        for filters in [64, 128, 256, 512]:
            x = BackboneNetwork._residual_block(x, filters)
            x = BackboneNetwork._residual_block(x, filters)
            if filters < 512:
                x = layers.Conv2D(filters * 2, 1, strides=2)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x, name='backbone')
    
    @staticmethod
    def _residual_block(x, filters: int):
        """Create a residual block for use in residual connections."""
        shortcut = x
        
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1)(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

class ROIHead:
    """ROI head with separate classification (what is it) and regression (where is it) branches. This design allows for more specialized learning in each branch, which can improve overall detection performance."""
    
    @staticmethod
    def create_roi_head(input_shape: Tuple[int, int, int], 
                       num_classes: int, 
                       config: DetectionConfig) -> tf.keras.Model:
        """Create ROI head with classification and bounding box regression."""
        inputs = layers.Input(shape=input_shape)
        
        # Shared feature extraction
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layers
        shared = layers.Dense(512, activation='relu')(x)
        shared = layers.Dropout(0.5)(shared)
        shared = layers.Dense(256, activation='relu')(shared)
        shared = layers.Dropout(0.3)(shared)
        
        # Classification branch
        cls_output = layers.Dense(num_classes, activation='softmax', name='classification')(shared)
        
        # Bounding box regression branch (4 coordinates: dx, dy, dw, dh)
        bbox_output = layers.Dense(4, activation='linear', name='bbox_regression')(shared)
        
        model = tf.keras.Model(inputs=inputs, outputs=[cls_output, bbox_output], name='roi_head')
        
        # Compile with multiple losses
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss={
                'classification': 'categorical_crossentropy',
                'bbox_regression': 'mse'
            },
            loss_weights={'classification': 1.0, 'bbox_regression': 0.5},
            metrics={
                'classification': ['accuracy'],
                'bbox_regression': ['mae']
            }
        )
        
        return model

class ImageAnalyzer:
    """Utility class for analyzing image properties."""
    
    @staticmethod
    def extract_dominant_colors(image: np.ndarray, n_colors: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from an image using K-means clustering. Depending on image quality, there may be tuning required to ensure the colors are representative of the image. I've included an image quality assessment function below that produces metadata identifiers that could be used for future applications. This can be applied at the model level as an iteration of this py file or in the warehouse after the model runs to determine whether the dataset is suitable for additional inference"""
        # Reshape image to be a list of pixels
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            # Convert grayscale to RGB if needed
            pixels = np.stack([image.flatten()] * 3, axis=1)
        
        # Handle edge cases
        if pixels.shape[0] == 0:
            return [(128, 128, 128)] * n_colors  # Default gray
        
        # Ensure we don't ask for more colors than pixels
        n_colors = min(n_colors, pixels.shape[0])
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Convert to integers and ensure valid RGB values
            colors = np.clip(colors, 0, 255).astype(int)
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logging.warning(f"Color extraction failed: {e}")
            return [(128, 128, 128)] * n_colors
    
    @staticmethod
    def assess_photo_quality(image: np.ndarray) -> float:
        """Assess photo quality using multiple metrics."""
        try:
            # Convert to grayscale for some calculations
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 1. Sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
            
            # 2. Contrast using standard deviation
            contrast_score = min(gray.std() / 60.0, 1.0)  # Normalize
            
            # 3. Brightness assessment (avoid too dark or too bright)
            mean_brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            
            # 4. Noise assessment (inverse of high frequency content)
            high_freq = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
            noise_level = high_freq.std() / 30.0
            noise_score = max(0.0, 1.0 - noise_level)
            
            # Combined quality score (weighted average)
            quality_score = (
                0.3 * sharpness_score +
                0.25 * contrast_score +
                0.25 * brightness_score +
                0.2 * noise_score
            )
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logging.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    @staticmethod
    def calculate_size_percentage(roi_area: int, total_image_area: int) -> float:
        """Calculate what percentage of the total image the ROI occupies."""
        if total_image_area == 0:
            return 0.0
        return (roi_area / total_image_area) * 100.0
    
    @staticmethod
    def get_color_description(colors: List[Tuple[int, int, int]]) -> List[str]:
        """Convert RGB colors to human-readable descriptions."""
        color_names = []
        
        for r, g, b in colors:
            # Simple color classification
            if r > 200 and g > 200 and b > 200:
                color_names.append("white")
            elif r < 50 and g < 50 and b < 50:
                color_names.append("black")
            elif r > g and r > b:
                if r > 150:
                    color_names.append("red")
                else:
                    color_names.append("dark_red")
            elif g > r and g > b:
                if g > 150:
                    color_names.append("green")
                else:
                    color_names.append("dark_green")
            elif b > r and b > g:
                if b > 150:
                    color_names.append("blue")
                else:
                    color_names.append("dark_blue")
            elif r > 150 and g > 150 and b < 100:
                color_names.append("yellow")
            elif r > 150 and g < 100 and b > 150:
                color_names.append("purple")
            elif r > 150 and g > 100 and b < 100:
                color_names.append("orange")
            elif abs(r - g) < 30 and abs(g - b) < 30:
                if r > 150:
                    color_names.append("light_gray")
                elif r > 100:
                    color_names.append("gray")
                else:
                    color_names.append("dark_gray")
            else:
                color_names.append("mixed")
        
        return color_names
    
    """Non-Maximum Suppression utilities. This is a common technique used in object detection to filter out overlapping bounding boxes based on their Intersection over Union (IoU) scores. The IoU score is calculated as the area of overlap between two bounding boxes divided by the area of their union. If the IoU score exceeds a certain threshold, one of the boxes is suppressed."""
    
    @staticmethod
    def iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def apply_nms(boxes: List[BoundingBox], 
                  confidence_threshold: float,
                  nms_threshold: float) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression."""
        # Filter by confidence
        filtered_boxes = [box for box in boxes if box.confidence >= confidence_threshold]
        
        if not filtered_boxes:
            return []
        
        # Sort by confidence (descending)
        filtered_boxes.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply NMS
        kept_boxes = []
        while filtered_boxes:
            current_box = filtered_boxes.pop(0)
            kept_boxes.append(current_box)
            
            # Remove boxes with high IoU
            filtered_boxes = [
                box for box in filtered_boxes
                if NonMaxSuppression.iou(current_box, box) < nms_threshold
            ]
        
        return kept_boxes

class EnhancedObjectDetectionPipeline:
    """Complete object detection pipeline with function calls for building models, extracting features, generating proposals, and processing detections."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.backbone = None
        self.roi_head = None
        self.storage = LocalStorageSimulator()
        self.class_names = {
            0: "Background",
            1: "FlowChart", 
            2: "NetworkDiagram",
            3: "Timeline"
        }
    
    def build_models(self, input_shape: Tuple[int, int, int]):
        """Build the backbone and ROI head models."""
        logger.info("Building models...")
        
        # Build backbone
        self.backbone = BackboneNetwork.create_resnet_backbone(input_shape)
        logger.info(f"Backbone built with input shape: {input_shape}")
        
        # Build ROI head
        self.roi_head = ROIHead.create_roi_head(
            self.config.roi_fixed_size, 
            self.config.num_classes,
            self.config
        )
        logger.info("ROI head built")
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using backbone network."""
        if self.backbone is None:
            raise ValueError("Backbone not initialized. Call build_models first.")
        
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        features = self.backbone.predict(image, verbose=0)
        return features
    
    def generate_roi_proposals(self, feature_map_shape: Tuple[int, int, int],
                              original_image_shape: Tuple[int, int, int],
                              num_proposals: int = 50) -> List[BoundingBox]:
        """Generate ROI proposals with realistic anchor-like generation."""
        proposals = []
        
        # Scale factors from feature map to original image
        scale_h = original_image_shape[0] / feature_map_shape[0]
        scale_w = original_image_shape[1] / feature_map_shape[1]
        
        # Multiple scales and aspect ratios
        scales = [0.5, 1.0, 2.0]
        aspect_ratios = [0.5, 1.0, 2.0]
        
        for _ in range(num_proposals):
            # Random center point on feature map
            center_y = np.random.randint(0, feature_map_shape[0])
            center_x = np.random.randint(0, feature_map_shape[1])
            
            # Random scale and aspect ratio
            scale = np.random.choice(scales)
            aspect_ratio = np.random.choice(aspect_ratios)
            
            # Calculate box dimensions
            base_size = 64 * scale
            width = int(base_size * np.sqrt(aspect_ratio))
            height = int(base_size / np.sqrt(aspect_ratio))
            
            # Convert to original image coordinates
            x1 = max(0, int((center_x * scale_w) - width // 2))
            y1 = max(0, int((center_y * scale_h) - height // 2))
            x2 = min(original_image_shape[1], x1 + width)
            y2 = min(original_image_shape[0], y1 + height)
            
            if x2 > x1 and y2 > y1:  # Valid box
                confidence = np.random.beta(2, 5)  # Skewed towards lower values
                class_id = np.random.randint(0, self.config.num_classes)
                
                proposals.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self.class_names[class_id]
                ))
        
        return proposals
    
    def roi_pooling(self, feature_map: np.ndarray, 
                    proposals: List[BoundingBox],
                    original_image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Simulate ROI pooling/align operation."""
        pooled_features = []
        
        # Scale factors
        scale_h = feature_map.shape[1] / original_image_shape[0]
        scale_w = feature_map.shape[2] / original_image_shape[1]
        
        for proposal in proposals:
            # Map proposal coordinates to feature map
            feat_x1 = int(proposal.x1 * scale_w)
            feat_y1 = int(proposal.y1 * scale_h)
            feat_x2 = int(proposal.x2 * scale_w)
            feat_y2 = int(proposal.y2 * scale_h)
            
            # Ensure valid coordinates
            feat_x1 = max(0, min(feat_x1, feature_map.shape[2] - 1))
            feat_y1 = max(0, min(feat_y1, feature_map.shape[1] - 1))
            feat_x2 = max(feat_x1 + 1, min(feat_x2, feature_map.shape[2]))
            feat_y2 = max(feat_y1 + 1, min(feat_y2, feature_map.shape[1]))
            
            # Extract and resize ROI features
            roi_features = feature_map[0, feat_y1:feat_y2, feat_x1:feat_x2, :]
            
            # Resize to fixed size (simplified - in practice use tf.image.resize)
            resized_features = np.random.rand(*self.config.roi_fixed_size)  # Placeholder
            pooled_features.append(resized_features)
        
        return np.array(pooled_features)
    
    def process_detections(self, pooled_rois: np.ndarray,
                          proposals: List[BoundingBox]) -> List[BoundingBox]:
        """Process ROIs through classification head and apply NMS."""
        if self.roi_head is None:
            raise ValueError("ROI head not initialized. Call build_models first.")
        
        # Get predictions
        cls_predictions, bbox_predictions = self.roi_head.predict(pooled_rois, verbose=0)
        
        # Process each detection
        detections = []
        for i, (cls_pred, bbox_pred, proposal) in enumerate(zip(cls_predictions, bbox_predictions, proposals)):
            # Get class with highest confidence
            class_id = np.argmax(cls_pred)
            confidence = cls_pred[class_id]
            
            # Apply bounding box regression (simplified)
            dx, dy, dw, dh = bbox_pred
            refined_x1 = proposal.x1 + dx
            refined_y1 = proposal.y1 + dy
            refined_x2 = refined_x1 + proposal.width + dw
            refined_y2 = refined_y1 + proposal.height + dh
            
            detection = BoundingBox(
                x1=int(refined_x1), y1=int(refined_y1),
                x2=int(refined_x2), y2=int(refined_y2),
                confidence=confidence,
                class_id=class_id,
                class_name=self.class_names[class_id]
            )
            detections.append(detection)
        
        # Apply Non-Maximum Suppression
        final_detections = NonMaxSuppression.apply_nms(
            detections,
            self.config.confidence_threshold,
            self.config.nms_threshold
        )
        
        return final_detections[:self.config.max_detections]
    
    def route_detections(self, detections: List[BoundingBox], 
                        original_image: np.ndarray):
        """Route classified detections to storage with comprehensive metadata."""
        total_image_area = original_image.shape[0] * original_image.shape[1]
        
        for detection in detections:
            # Extract ROI from original image
            roi_image = original_image[
                detection.y1:detection.y2,
                detection.x1:detection.x2
            ]
            
            # Analyze ROI properties
            roi_analysis = self.analyze_roi_properties(roi_image, detection, total_image_area)
            
            # Create comprehensive metadata
            metadata = {
                # Basic detection info
                'class_name': detection.class_name,
                'class_id': detection.class_id,
                'confidence': float(detection.confidence),
                
                # Spatial information
                'bbox': {
                    'x1': detection.x1, 'y1': detection.y1,
                    'x2': detection.x2, 'y2': detection.y2,
                    'width': detection.width, 'height': detection.height
                },
                'size_analysis': {
                    'area_pixels': roi_analysis['roi_area_pixels'],
                    'percentage_of_image': roi_analysis['size_percentage'],
                    'size_category': roi_analysis['size_category']
                },
                
                # Color analysis
                'color_analysis': {
                    'dominant_colors_rgb': roi_analysis['dominant_colors_rgb'],
                    'dominant_colors_names': roi_analysis['dominant_colors_names'],
                    'primary_color': roi_analysis['dominant_colors_names'][0] if roi_analysis['dominant_colors_names'] else "unknown"
                },
                
                # Quality assessment
                'quality_analysis': {
                    'quality_score': roi_analysis['photo_quality_score'],
                    'quality_rating': roi_analysis['quality_rating'],
                    'is_high_quality': roi_analysis['photo_quality_score'] >= 0.6
                },
                
                # Image context
                'image_context': {
                    'original_image_dimensions': {
                        'height': original_image.shape[0],
                        'width': original_image.shape[1],
                        'channels': original_image.shape[2] if len(original_image.shape) > 2 else 1
                    },
                    'total_image_area': total_image_area,
                    'roi_to_image_ratio': roi_analysis['size_percentage'] / 100.0
                },
                
                # Processing metadata
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'roi_shape': roi_image.shape,
                    'detection_pipeline_version': "2.0"
                }
            }
            
            # Save to storage
            roi_id = self.storage.save_roi(roi_image, metadata)
            
            # Enhanced logging with new metadata
            logger.info(f"Routed detection {roi_id}: {detection.class_name} "
                       f"(conf: {detection.confidence:.3f}, "
                       f"quality: {roi_analysis['quality_rating']}, "
                       f"size: {roi_analysis['size_category']} - {roi_analysis['size_percentage']:.1f}%, "
                       f"colors: {', '.join(roi_analysis['dominant_colors_names'][:2])})")
    
    def run_pipeline(self, input_image: np.ndarray) -> List[BoundingBox]:
        """Run the complete detection pipeline."""
        logger.info(f"Starting detection pipeline on image shape: {input_image.shape}")
        
        # 1. Feature extraction
        features = self.extract_features(input_image)
        logger.info(f"Extracted features shape: {features.shape}")
        
        # 2. Generate proposals
        proposals = self.generate_roi_proposals(
            features.shape[1:], 
            input_image.shape,
            num_proposals=100
        )
        logger.info(f"Generated {len(proposals)} proposals")
        
        # 3. ROI pooling
        pooled_rois = self.roi_pooling(features, proposals, input_image.shape)
        logger.info(f"Pooled ROIs shape: {pooled_rois.shape}")
        
        # 4. Detection processing
        final_detections = self.process_detections(pooled_rois, proposals)
        logger.info(f"Final detections after NMS: {len(final_detections)}")
        
        # 5. Route detections
        self.route_detections(final_detections, input_image)
        
        return final_detections

def main():
    """Main execution function."""
    # Configuration
    config = DetectionConfig(
        roi_fixed_size=(64, 64, 3),
        num_classes=4,
        confidence_threshold=0.3,
        nms_threshold=0.5
    )
    
    # Create pipeline
    pipeline = EnhancedObjectDetectionPipeline(config)
    
    # Simulate variable input size
    input_height = np.random.randint(256, 1024)
    input_width = np.random.randint(256, 1024)
    input_shape = (input_height, input_width, 3)
    
    # Build models
    pipeline.build_models(input_shape)
    
    # Create dummy input image
    dummy_image = np.random.rand(*input_shape).astype(np.float32)
    
    # Train ROI head with dummy data (simplified)
    logger.info("Training ROI head...")
    dummy_rois = np.random.rand(50, *config.roi_fixed_size).astype(np.float32)
    dummy_cls_labels = tf.keras.utils.to_categorical(
        np.random.randint(0, config.num_classes, 50), 
        config.num_classes
    )
    dummy_bbox_labels = np.random.randn(50, 4).astype(np.float32)
    
    pipeline.roi_head.fit(
        dummy_rois,
        {'classification': dummy_cls_labels, 'bbox_regression': dummy_bbox_labels},
        epochs=3,
        batch_size=config.batch_size,
        verbose=1
    )
    
    # Run detection pipeline
    detections = pipeline.run_pipeline(dummy_image)
    
    # Print results with enhanced metadata
    logger.info("Detection Results:")
    for i, detection in enumerate(detections):
        logger.info(f"  {i+1}. {detection.class_name} "
                   f"(conf: {detection.confidence:.3f}, "
                   f"box: [{detection.x1},{detection.y1},{detection.x2},{detection.y2}], "
                   f"quality: {detection.photo_quality_score:.2f}, "
                   f"size: {detection.size_percentage:.1f}% of image)")

if __name__ == "__main__":
    main()