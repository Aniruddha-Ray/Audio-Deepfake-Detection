# Real-Time Deepfake Voice Detection

**Event:** PSB Hackathon Series 2026
**Team:** BEing Techies
**Institution:** Indian Institute of Engineering Science and Technology, Shibpur

## Team Members
* Antarikshya Mitra
* Aniruddha Ray
* Subhadip Bhunia

## Problem
DeepFake audio technology has evolved rapidly, enabling near-perfect voice cloning with just seconds of sample audio. This poses critical threats to voice-based authentication systems, financial fraud through impersonation calls, and the spread of misinformation. In 2023 alone, voice fraud losses exceeded $25 billion globally. Traditional detection methods fail against sophisticated AI-generated voices, creating an urgent need for robust, real-time deepfake voice detection solutions.

## Proposed Solution
Designing and deploying a real-time audio forensics system that detects AI-generated (deepfake) voices within the first 10 seconds of a live call by analyzing spectrogram and MFCC-based features using a hybrid CNN-LSTM and SVM ensemble model. This is integrated with a streaming ETL pipeline and scalable CI/CD-enabled deployment in order to prevent voice-based financial fraud and strengthen authentication systems.

## System Architecture & Workflow
* **Real-Time Data Ingestion:** Captures live audio via VoIP/SIP APIs and streams it in small manageable chunks (0.5-1 sec) using a high-speed Kafka streaming backbone.
* **Preprocessing & Transformation:** Resamples audio to 16 kHz, applies noise reduction and Voice Activity Detection (VAD), and forms 2-second segments using a 10-second buffer.
* **Feature Extraction (Parallel):** Extracts Log-Mel Spectrograms alongside MFCC and Spectral Features.
* **Hybrid Model:** Utilizes a deep learning branch (CNN + LSTM/Transformer via PyTorch) and a classical machine learning branch (SVM + MFCC Features via scikit-learn/LIBSVM).
* **Ensemble Layer:** Combines outputs using weighted fusion or stacking (Meta Model: Logistic Regression/XGBoost) to produce a final probability score.
* **Risk Engine:** Classifies the final probability into High Risk (Block), Medium Risk (Verify), or Low Risk (Allow).
* **Deployment & CI/CD:** Deployed via FastAPI, Docker, and Kubernetes, orchestrated with automated GitHub Actions on cloud platforms (AWS/GCP).
* **Monitoring & Output:** Utilizes Prometheus for metrics tracking and Grafana for live dashboards, outputting real-time detection alerts within 10 seconds.

## Performance Metrics
Our deepfake voice detection system demonstrates strong real-world performance across critical metrics, showing high robustness against noisy environments and unseen deepfake variations:
* **Accuracy:** 99.59% on benchmark datasets.
* **Equal Error Rate (EER):** 0.43%.
* **False Positive Rate (FPR):** 0.0043.
* **Recall:** 0.9969.
* **Latency:** ~0.0085 seconds per audio segment (for each input of 300 frames).

## Conclusion & Novelty
* **Novelty:** Transforms deepfake detection from an offline model into a deployable, real-time fraud prevention product for banking and secure communication systems.
* **Innovation:** Leverages cutting-edge AI and machine learning algorithms to identify synthetic audio with unprecedented precision.
* **Speed:** Detects deepfake voices within the first 10 seconds of a live call using a continuous streaming ETL pipeline.
* **Reliability:** The system is production-ready, featuring a microservices architecture, CI/CD integration, and integrated monitoring for explainability and performance tracking.

## References
* "Audio Deepfake Detection Using Deep Learning" - Ousama A. Shaaban, Remzi Yildirim.
* "Deepfake Audio Detection Using Spectrogram-based Feature and Ensemble of Deep Learning Models" - Lam Pham, Phat Lam, Truong Nguyen, Huyen Nguyen, Alexander Schindler.
* "Deepfake Audio Detection via MFCC features using Machine Learning" - Ameer Hamza, Abdul Rehman Javed, Farkhund Iqbal, Natalia Kryvinska, Ahmad S. Almadhor, Zunera Jalil, Rouba Borghol.
* "Deep Fake Audio Detection" - Dr Anupama Kumar, Pranathi V, Shreyas R.
