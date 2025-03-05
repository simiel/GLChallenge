# CRM Admin Dashboard

A modern CRM dashboard built with Next.js 14, featuring customer churn prediction capabilities and comprehensive customer management.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## API Routes

### Backend API (Production)

Base URL: `https://glchallenge.onrender.com`

#### Prediction API
- **POST** `/api/predict`
  ```typescript
  // Request Body
  {
    TransactionCount: number,
    AverageTransactionAmount: number,
    TotalTransactionAmount: number,
    TransactionAmountStd: number,
    Age: number,
    AccountBalance: number,
    DaysSinceLastTransaction: number,
    CustomerTenure: number,
    TransactionsPerMonth: number,
    Gender: "M" | "F",
    Location: string
  }
  
  // Response
  {
    churn_probability: number // Between 0 and 1
  }
  ```

### Frontend API Routes

#### Authentication
- **POST** `/api/auth/[...nextauth]`
  - Handles authentication using NextAuth.js
  - Supports credentials provider

## Authentication

### Demo Login Credentials
```
Email: admin@example.com
Password: password123
```

## Recent Updates & Progress

1. **Authentication System**
   - Implemented secure login with NextAuth.js
   - Added protected routes and session management

2. **Churn Prediction Feature**
   - Integrated ML model API for churn predictions
   - Created user-friendly prediction form with validation
   - Added real-time prediction results with risk assessment

3. **UI Components**
   - Implemented form components using shadcn/ui
   - Added toast notifications for user feedback
   - Created responsive layout with mobile support

4. **Data Management**
   - Set up Prisma ORM for database operations
   - Created database schema for users and predictions
   - Implemented data seeding for demo accounts

## Tech Stack

- **Frontend**: Next.js 14, React, TypeScript
- **UI**: Tailwind CSS, shadcn/ui
- **Forms**: React Hook Form, Zod
- **Authentication**: NextAuth.js
- **Database**: SQLite (Development), Prisma ORM
- **API Integration**: REST APIs with fetch

## Project Structure

```
crm-admin/
├── app/
│   ├── (auth)/
│   │   └── login/
│   └── (dashboard)/
│       └── dashboard/
│           └── predictions/
├── components/
│   └── ui/
├── lib/
├── prisma/
└── public/
```

## Environment Variables

Create a `.env` file in the root directory with:

```env
DATABASE_URL="file:./dev.db"
NEXTAUTH_SECRET="your-secret-here"
NEXTAUTH_URL="http://localhost:3000"
```

## Development Status

- [x] Basic authentication
- [x] Churn prediction integration
- [x] Form validation
- [x] UI components
- [ ] Customer management
- [ ] Analytics dashboard
- [ ] User settings
