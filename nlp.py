import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simulate Big Data - Stream Metrics + Feedback
def simulate_stream_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'bitrate': np.random.randint(1000, 8000, size=n),
        'latency': np.random.uniform(0.1, 3.0, size=n),
        'frame_drops': np.random.randint(0, 20, size=n),
        'feedback_text': np.random.choice([
            "Great stream!", "Laggy and poor quality", "Loved the gameplay!", 
            "Buffering too much", "Smooth and clear", "Terrible experience"
        ], size=n),
        'satisfaction': np.random.choice([0, 1], size=n)  # 0 = unsatisfied, 1 = satisfied
    })
    return data

# Step 1: Sentiment Analysis using BERT
def analyze_sentiment(data):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = [sentiment_analyzer(text)[0]['label'] for text in data['feedback_text']]
    data['sentiment'] = [1 if s == 'POSITIVE' else 0 for s in sentiments]
    return data

# Step 2: DNN Model for QoS Satisfaction Classification
def train_dnn_model(data):
    features = data[['bitrate', 'latency', 'frame_drops', 'sentiment']]
    labels = data['satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, input_dim=features.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    return model

# Step 3: Dynamic QoS Management
def adjust_qos(row):
    if row['sentiment'] == 0 or row['satisfaction'] == 0:
        return 'Increase Bitrate / Reduce Latency'
    return 'Maintain QoS'

# Main Flow
if __name__ == "__main__":
    stream_data = simulate_stream_data()
    stream_data = analyze_sentiment(stream_data)
    model = train_dnn_model(stream_data)
    stream_data['qos_action'] = stream_data.apply(adjust_qos, axis=1)

    print(stream_data[['bitrate', 'latency', 'frame_drops', 'feedback_text', 'sentiment', 'satisfaction', 'qos_action']].head())
