🌍 Disaster Tweets Classifier
📌 Overview

This project classifies tweets as disaster-related (1) or non-disaster (0) using Logistic Regression with TF-IDF.
It is part of the Climate Risk & Disaster Management theme to help in early detection and situational awareness during disasters.

⚙️ Workflow

Data Preprocessing → clean tweets (remove links, hashtags, special characters).

EDA → class distribution, tweet length, wordclouds.

Feature Extraction → TF-IDF (top 5000 features).

Model Training → Logistic Regression.

Evaluation → Accuracy (~81%), Confusion Matrix, ROC Curve (AUC ~0.85).
