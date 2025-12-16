# Etioplast-Quantification

## This project focuses on the automated quantification of subcellular structures within etioplasts using deep learning and web technologies.

# Etioplast Analysis & Quantification Tool 🌿🔬

A Django-based computer vision application designed for the automated segmentation, quantification, and biological analysis of plant cell ultrastructures (specifically Etioplasts) using **YOLOv8** and **Generative AI**.

## 🚀 Features

* **Automated Segmentation**: Uses custom-trained YOLOv8 models to detect:
    * Etioplasts
    * Prolamellar Bodies (PLB)
    * Prothylakoids
    * Plastoglobules
    * Starch Grains
* **Geometric Quantification**: Calculates precise metrics including:
    * Total Area ($\mu m^2$)
    * Total Length ($\mu m$)
    * Average Diameter ($\mu m$)
* **Hierarchical Analysis**: Establishes parent-child relationships between organelles (e.g., determining which plastoglobules are inside an etioplast).
* **Visualizations**: Generates:
    * Annotated overlays with bounding boxes and contours.
    * Individual binary masks for specific organelles.
* **Generative AI Summary**: Integrates with an LLM API to provide biological interpretations of the quantitative data.
* **REST API**: Provides endpoints for processing single images or entire folders.

## 🛠️ Tech Stack

* **Framework**: Django (Python)
* **Computer Vision**: Ultralytics YOLOv8, OpenCV, Scikit-Image, NumPy
* **AI/LLM**: Custom Generative AI integration (via API)
* **Data Processing**: Pandas (implied for CSVs), Python standard libraries

## 📋 Prerequisites

* Python 3.8+
* CUDA-enabled GPU (recommended for faster YOLO inference)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd etioplast-analysis
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    *(Note: Ensure you have a `requirements.txt`. Based on the code, you will need the following)*
    ```bash
    pip install django ultralytics opencv-python-headless numpy scikit-image requests python-dotenv
    ```

4.  **Environment Setup**
    Create a `.env` file in the root directory and configure the following variables:
    ```env
    # Django Settings
    DEBUG=True
    SECRET_KEY=your_django_secret_key

    # Model Configuration
    MODEL_PATH=path/to/your/yolov8-seg.pt
    PX_PER_UM=344.8  # Pixels per micrometer calibration (example)

    # Generative AI Configuration
    API_KEY=your_llm_api_key
    API_URL=your_llm_api_endpoint
    LLM_MODEL=your_preferred_model_name
    ```

5.  **Database Migrations**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

## 🚀 Usage

### Starting the Server
```bash
python manage.py runserver