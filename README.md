# Scholar App Backend

A Python Flask backend API that powers the Scholar App scholarship matching platform. It provides ML-based admission prediction for 9 Pakistani universities, fee structure scraping, university event aggregation, and data processing -- serving as the intelligence layer behind the Scholar App Frontend.

## Features

- ML-based admission prediction models trained on historical data for 9 Pakistani universities
- REST API endpoints for scholarship matching and university recommendations
- Automated fee structure scraping from university websites using BeautifulSoup and Selenium
- University events aggregation and listing
- Data preprocessing and feature engineering pipelines
- Pre-trained scikit-learn and TensorFlow models for prediction accuracy
- Structured data storage and retrieval with pandas

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Flask | Lightweight web framework for REST API |
| scikit-learn | Traditional ML models for admission prediction |
| TensorFlow | Deep learning models for prediction |
| pandas | Data manipulation and preprocessing |
| BeautifulSoup | HTML parsing for web scraping |
| Selenium | Browser automation for dynamic page scraping |

## Getting Started

### Prerequisites

- Python 3.9+ installed
- pip package manager
- Chrome browser and ChromeDriver (for Selenium scraping)

### Installation

```bash
git clone https://github.com/Faze789/scholar_app.git
cd scholar_app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Set up environment variables or a `.env` file with any required API keys and database connection strings.

### Running the Server

```bash
flask run
```

The API will be available at `http://localhost:5000`.
