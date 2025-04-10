# Brain Tumor Detection System

An AI-powered brain tumor detection system using deep learning to analyze MRI scans.

## Features

- Real-time tumor detection from MRI scans
- Multi-class tumor classification (Glioma, Meningioma, Pituitary)
- Interactive visualization of tumor regions
- Detailed analysis with bounding boxes and tumor outlines
- Comprehensive statistics and model performance metrics
- Educational content about brain tumors

## Project Structure

```
brain-tumor-detection/
├── app/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py
│   │   ├── model_utils.py
│   │   └── visualization.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── landing.py
│   │   ├── detection.py
│   │   └── stats.py
│   └── config.py
├── models/
│   └── model.h5
├── static/
│   └── images/
├── tests/
│   └── test_model.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd brain-tumor-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the trained model:
   - Put your trained model file (`model.h5`) in the `models/` directory

5. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t brain-tumor-detection .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 brain-tumor-detection
   ```

The application will be available at `http://localhost:8501`

## Model Information

- Architecture: CNN with transfer learning
- Input size: 128x128x3
- Classes: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- Accuracy: 95%
- Training dataset: 3000+ MRI scans

## Environment Variables

Create a `.env` file with the following variables:
```
MODEL_PATH=models/model.h5
DEBUG=False
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle Brain Tumor Dataset]
- Medical imaging references
- Contributors and maintainers 