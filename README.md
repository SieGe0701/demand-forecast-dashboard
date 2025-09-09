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

#### Add Your Data
- Place your dataset (e.g., `data.csv`) in the `data/` folder.
- You can use any tabular format (CSV recommended).

#### Dashboard
- Start the dashboard: `python dashboard/app.py` or `streamlit run dashboard/app.py`
- Upload your dataset via the dashboard interface or use the file in `data/`.
- Select the target column and configure feature engineering options.
- Train the model and view predictions directly in the dashboard.

#### API
- Start the API: `uvicorn api.main:app --reload --port 8000`
- Send POST requests to `/predict` with your data as JSON for batch predictions.

#### Scripts
- Use the functions in `src/` to preprocess, train, and predict programmatically.
- Example:
	```python
	from src.data_preprocessing import load_data, clean_data, transform_data
	from src.train_model import train_and_save_model
	from src.predict import load_model, predict

	df = load_data('data/data.csv')
	df = clean_data(df)
	df = transform_data(df, target_col='target')
	model, mse = train_and_save_model(df, 'target')
	preds = predict(model, df.drop(columns=['target']))
	```

#### Access Dashboard
- Open your browser and go to `http://localhost:8501` after starting the dashboard.

#### Access API
- Send requests to `http://localhost:8000/predict` after starting the API.

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

