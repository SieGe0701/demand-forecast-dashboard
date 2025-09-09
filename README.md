## Demand Forecast Dashboard

An end-to-end demand forecasting dashboard with CI/CD (MLOps) pipeline. This project includes data preprocessing, model training, prediction API, and an interactive dashboard for visualization. Automated testing and deployment are integrated using CI/CD best practices.

### Features
- Data preprocessing and cleaning
- Model training and evaluation
- REST API for predictions
- Interactive dashboard (e.g., Streamlit)
- Automated testing (pytest)
- Dockerized deployment
- CI/CD pipeline for automated build, test, and deploy

### Prerequisites
- Python 3.8+
- Docker
- Git
- Recommended: VS Code

### Setup
1. Clone the repository:
	```powershell
	git clone https://github.com/SieGe0701/demand-forecast-dashboard.git
	cd demand-forecast-dashboard
	```
2. Install dependencies:
	```powershell
	pip install -r requirements.txt
	```
3. Run tests:
	```powershell
	pytest tests/
	```
4. Build and run with Docker:
	```powershell
	docker build -t demand-forecast-dashboard .
	docker run -p 8501:8501 demand-forecast-dashboard
	```

### Usage
- Start the API: `python api/main.py`
- Start the dashboard: `python dashboard/app.py`
- Access dashboard at `http://localhost:8501`

### CI/CD Workflow
CI/CD is set up to automatically:
- Run tests on every push
- Build Docker image
- Deploy to cloud (e.g., Azure, AWS, GCP)

See `.github/workflows/` for pipeline configuration (to be added).

### Project Structure
```
├── api/                # FastAPI or Flask REST API
├── dashboard/          # Streamlit or Dash dashboard
├── src/                # Data preprocessing, model training, prediction
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerization
├── README.md           # Project documentation
```

### Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

