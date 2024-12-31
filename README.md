# Project Overview

## Face Recognition
- **DeepFace Library**: Utilizes the DeepFace library with a pre-trained VGG-Face model for extracting facial embeddings.
- **Embeddings Management**: Stores embeddings for registered individuals and recognizes new faces based on a threshold distance.
- **Presence Tracking**: Automatically updates a presence file to track recognized individuals in the system.

## Clothing-Based Recognition
- **Feature Extraction**: Employs MobileNetV2 for extracting clothing features from images.
- **Similarity Matching**: Matches clothing features with stored profiles using cosine similarity to identify individuals when facial recognition is insufficient.
- **Dynamic Updates**: Updates stored clothing features dynamically for improved future recognition.

## YOLO-Based Facial Recognition
- **YOLO Model**: Implements facial recognition using the YOLO architecture in the `yolo` folder.
- **Comparison**: Provides a comparative analysis of the YOLO-based approach versus the DeepFace and MobileNetV2 methods.

## Blockchain Integration
- **Blockchain Module**: Includes a tested blockchain module in the `blockchain` folder.
- **Traceability**: Ensures secure and traceable management of recognition data for enhanced reliability and auditability.
