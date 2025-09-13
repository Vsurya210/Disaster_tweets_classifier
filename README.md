🌍 Disaster Tweets Classifier
📌 Overview
During disasters such as floods, earthquakes, or wildfires, people often share information on Twitter in real time.
However, not all tweets are relevant — some may be jokes, opinions, or unrelated chatter.
This creates a challenge for disaster management teams to quickly filter useful tweets from noise.

The Disaster Tweets Classifier solves this problem by using Natural Language Processing (NLP) and Machine Learning:
      It analyzes tweet text and predicts whether it is disaster-related (1) or non-disaster (0).
      The model is trained using TF-IDF features and Logistic Regression.
      This system can be integrated into early warning systems, social media monitoring dashboards, or real-time disaster response tools.

⚙️ Workflow

Data Preprocessing → clean tweets (remove links, hashtags, special characters).

EDA → class distribution, tweet length, wordclouds.

Feature Extraction → TF-IDF (top 5000 features).

Model Training → Logistic Regression.

Evaluation → Accuracy (~81%), Confusion Matrix, ROC Curve (AUC ~0.85).
