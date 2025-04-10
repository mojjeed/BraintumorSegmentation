# Brain Tumor Detection App

This Streamlit application uses deep learning to detect and mark brain tumors in MRI scans.

## Features

- Upload MRI scans for tumor detection
- Automatic tumor detection using deep learning
- Visual marking of detected tumors
- Confidence score display
- Download marked images

## Local Development

1. Clone the repository:
```bash
git clone <your-repository-url>
cd brain-tumor-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run streamlit_app.py
```

## Deployment

### Deploying to Streamlit Cloud

1. Create a GitHub repository and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repository-url>
git push -u origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file path (streamlit_app.py)
6. Click "Deploy"

## Project Structure

```
brain-tumor-detection/
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Project dependencies
├── README.md                # Project documentation
│
├── models/                  # Directory for model files
│   └── model.h5            # Your trained model
│
└── .gitignore              # Git ignore file
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy
- Pillow

## License

[Your chosen license]