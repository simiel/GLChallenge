# AI-Driven Customer Churn Prediction Platform

A modern, full-stack application for predicting customer churn using AI and machine learning techniques. The platform features an interactive dashboard and is deployed on Cloudflare's edge network.

## Architecture

- **Frontend**: React + Shadcn UI for modern, responsive dashboard
- **Backend**: Python-based ML pipeline with ensemble models
- **API**: Cloudflare Workers for serverless endpoints
- **Deployment**: Cloudflare Pages (Frontend) + Cloudflare Workers (Backend)
- **Containerization**: Docker for consistent development and deployment

## Project Structure

```
.
├── frontend/           # React dashboard application
├── backend/           # Python ML pipeline
│   ├── data/         # Data processing scripts
│   ├── models/       # ML model implementations
│   └── api/          # API endpoints
├── workers/          # Cloudflare Workers code
└── docker/           # Docker configuration files
```

## Prerequisites

- Node.js 18+
- Python 3.9+
- Docker
- Cloudflare account
- Kaggle API credentials

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Set up the Python environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the development environment:
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

## Deployment

1. Deploy the frontend to Cloudflare Pages:
```bash
cd frontend
npm run build
# Deploy using Cloudflare Pages dashboard
```

2. Deploy the backend to Cloudflare Workers:
```bash
cd workers
wrangler deploy
```

## Features

- Interactive data visualization dashboard
- Real-time churn prediction
- Feature importance analysis
- Customer segmentation
- Automated model retraining
- Edge computing for low-latency predictions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License - see LICENSE file for details
