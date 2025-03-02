# AI-Driven CRM Analytics & Customer Churn Prediction

## Project Overview
This proof of concept demonstrates an AI-driven CRM analytics platform with customer churn prediction capabilities, built using Cloudflare's serverless infrastructure and AI-assisted development via Cursor.

### 🎯 Objective
Design and implement a containerized CRM analytics platform that leverages AI for customer churn prediction, utilizing Cloudflare's free-tier services and AI-assisted development practices.

### 🏗️ Technical Stack
- **Frontend**: React with Shadcn UI
- **Backend**: Cloudflare Workers
- **Hosting**: Cloudflare Pages
- **Development**: AI-assisted (Cursor)
- **Deployment**: Containerized, zero-cost architecture

### 🔑 Key Features
1. **Data Processing**
   - Kaggle dataset integration
   - Automated data ingestion pipeline
   - AI-driven feature engineering

2. **Machine Learning**
   - Ensemble churn prediction model
   - Advanced feature engineering
   - Model validation and metrics
   - Real-time prediction capabilities

3. **User Interface**
   - Interactive analytics dashboard
   - Data visualization components
   - Customer insights display
   - Responsive design

4. **Backend Services**
   - RESTful API endpoints
   - Serverless compute
   - Secure data handling
   - Scalable architecture

### ⏰ Timeline
- **Start Date**: February 28, 2025
- **Completion Deadline**: February 4, 2025 (0100 GMT)

### 📋 Project Requirements

#### Core Requirements
- Containerized application architecture
- AI-assisted development implementation
- Zero-cost deployment using Cloudflare
- Data ingestion and processing pipeline
- Machine learning model implementation
- Interactive frontend dashboard

#### Technical Requirements
- React-based frontend with Shadcn UI
- Cloudflare Workers for backend services
- Cloudflare Pages for hosting
- Ensemble ML model implementation
- Data visualization components
- API endpoint implementation

### 🚀 Getting Started
[To be added: Setup instructions]

### 📊 Progress Tracking
[To be added: Development progress]

### 📝 Documentation
[To be added: Detailed documentation]

### 🤝 Contributing
[To be added: Contribution guidelines]

### 📜 License
[To be added: License information]

### References
1. [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)
2. [React Official Documentation](https://react.dev/)
3. [Shadcn UI Components](https://ui.shadcn.com/)
4. [Machine Learning Model Development Guide](https://scikit-learn.org/stable/tutorial/index.html)
5. [Data Visualization with React](https://recharts.org/en-US/)
6. [Containerization Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)


### Directories

```
GLChallenge/
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   │   ├── dashboard/     # Dashboard-specific components
│   │   │   ├── charts/       # Data visualization components
│   │   │   └── common/       # Shared components
│   │   ├── hooks/            # Custom React hooks
│   │   ├── pages/            # Page components
│   │   ├── services/         # API service integrations
│   │   ├── styles/           # Global styles and themes
│   │   ├── types/           # TypeScript type definitions
│   │   └── utils/           # Helper functions and utilities
│   ├── public/              # Static assets
│   └── tests/              # Frontend tests
│
├── backend/                # Cloudflare Workers backend
│   ├── src/
│   │   ├── handlers/      # API route handlers
│   │   ├── middleware/    # Request/response middleware
│   │   ├── models/        # Data models
│   │   ├── services/      # Business logic
│   │   └── utils/         # Helper functions
│   └── tests/            # Backend tests
│
├── ml/                    # Machine Learning components
│   ├── models/           # Trained ML models
│   ├── notebooks/        # Jupyter notebooks for analysis
│   ├── preprocessing/    # Data preprocessing scripts
│   ├── training/        # Model training scripts
│   └── utils/           # ML utilities
│
├── data/                 # Data management
│   ├── raw/             # Original dataset
│   ├── processed/       # Processed dataset
│   └── schemas/         # Data validation schemas
│
├── infrastructure/       # Infrastructure as code
│   ├── docker/          # Docker configuration
│   └── cloudflare/      # Cloudflare configuration
│
├── docs/                # Documentation
│   ├── api/            # API documentation
│   ├── ml/             # ML model documentation
│   └── setup/          # Setup guides
│
└── scripts/            # Development and deployment scripts
```

### Key Files

```
GLChallenge/
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
├── package.json              # Project dependencies
├── tsconfig.json            # TypeScript configuration
├── docker-compose.yml       # Docker composition
├── .env.example            # Environment variables template
└── wrangler.toml          # Cloudflare Workers configuration
```
