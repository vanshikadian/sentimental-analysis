# Sentiment Analysis Web Application

## Overview

This project is a web application for sentiment analysis of textual reviews. It uses a machine learning model trained on a dataset of reviews to predict the sentiment of new reviews as Positive, Neutral, or Negative. Users can input individual text reviews directly through the web interface or upload a CSV file containing multiple reviews for batch processing.

## Features

- **Text Review Input**: Enter a text review directly through the web interface and get the sentiment prediction.
- **CSV File Upload**: Upload a CSV file containing reviews and download a processed CSV file with sentiment predictions.
- **Bootstrap Styling**: Clean and modern user interface using Bootstrap framework.


## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/sentiment-analysis-web-app.git
    cd sentiment-analysis-web-app
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK data**:
    ```python
    import nltk
    nltk.download('stopwords', download_dir='./nltk_data')
    nltk.download('wordnet', download_dir='./nltk_data')
    ```

5. **Run the Flask application**:
    ```bash
    export FLASK_ENV=development
    export FLASK_APP=app.py
    flask run
    ```

6. **Open your web browser** and go to `http://127.0.0.1:5000/` to access the web application.

## Usage

- **Enter a Review**: Input your review in the text box and click "Submit" to get the sentiment prediction.
- **Upload a CSV File**: Choose a CSV file containing reviews, upload it, and download the processed file with sentiment predictions.


Contributions are welcome! Please feel free to submit a Pull Request.





