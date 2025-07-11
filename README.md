# 🏋️ AI Sports Coach – Free Squat Feedback App

This is a fully automated AI web app that evaluates squat form from a video using computer vision (MediaPipe) and gives coaching feedback.

## 🚀 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/your-username/ai_sports_coach.git
cd ai_sports_coach
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the app:
```bash
streamlit run ai_sports_coach.py
```

## 🌐 Free Hosting (Optional)
Use [Streamlit Cloud](https://share.streamlit.io/) to deploy your app.

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud and connect your GitHub
3. Choose `ai_sports_coach.py` as entry point
4. Done!

## 📦 Tech Stack
- Pose detection: Google MediaPipe
- UI: Streamlit
- Feedback logic: Rule-based (extendable to LLM)
- Tracking: JSON + Matplotlib
