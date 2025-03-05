# Technical Report: CRM Admin Dashboard with Churn Prediction

## Project Overview

This report details the implementation of a CRM Admin Dashboard with integrated customer churn prediction capabilities. The system follows a modern architecture with clear separation between frontend, backend, and machine learning components.

## System Architecture

### 1. Machine Learning Component
- **Model**: Customer Churn Prediction
- **Deployment**: Hosted on Render.com
- **Endpoint**: `https://glchallenge.onrender.com/api/predict`

#### Model Features
- Transaction-based metrics
- Customer demographic data
- Behavioral patterns
- Historical usage data

#### Input Schema
```typescript
{
  TransactionCount: number,        // Total number of transactions
  AverageTransactionAmount: number, // Mean transaction value
  TotalTransactionAmount: number,   // Sum of all transactions
  TransactionAmountStd: number,     // Standard deviation of transaction amounts
  Age: number,                      // Customer age
  AccountBalance: number,           // Current account balance
  DaysSinceLastTransaction: number, // Days since last activity
  CustomerTenure: number,           // Length of customer relationship
  TransactionsPerMonth: number,     // Average monthly transactions
  Gender: "M" | "F",                // Customer gender
  Location: string                  // Customer location
}
```

#### Output Schema
```typescript
{
  churn_probability: number  // Probability between 0 and 1
}
```

### 2. Backend Implementation

#### Database Schema (Prisma)
```prisma
model User {
  id            String    @id @default(cuid())
  name          String?
  email         String    @unique
  password      String
  role          String    @default("user")
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
}

model Prediction {
  id                String   @id @default(cuid())
  userId            String
  user              User     @relation(fields: [userId], references: [id])
  transactionCount  Int
  avgAmount         Float
  totalAmount       Float
  amountStd         Float
  age               Int
  balance           Float
  lastTransaction   Int
  tenure            Int
  monthlyTx         Float
  gender            String
  location          String
  churnProbability  Float
  createdAt         DateTime @default(now())
}
```

#### API Routes

1. **Authentication Routes**
   - `POST /api/auth/[...nextauth]`
   - Handles user authentication using NextAuth.js
   - Implements credentials provider

2. **Prediction Routes**
   - `POST /api/predict`
   - Integrates with ML model
   - Handles data validation and transformation

### 3. Frontend Implementation

#### Tech Stack
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui components
- React Hook Form
- Zod validation

#### Key Components

1. **Authentication**
   - Login page with form validation
   - Protected route middleware
   - Session management

2. **Prediction Form**
   - Real-time validation
   - Type-safe form handling
   - Error handling and user feedback
   - Loading states and animations

3. **UI Components**
   - Form components (input, select, button)
   - Toast notifications
   - Loading spinners
   - Card layouts

#### State Management
- React Hook Form for form state
- NextAuth.js for authentication state
- Local state for UI components

## Deployment

### 1. ML Model Deployment
- Hosted on Render.com
- RESTful API endpoint
- CORS enabled for frontend access

### 2. Frontend Deployment
- Development: Local environment
- Production: Vercel (recommended)
- Environment variables configured

## Security Implementation

1. **Authentication**
   - Password hashing with bcrypt
   - JWT-based session management
   - Protected API routes

2. **Data Validation**
   - Input sanitization
   - Type checking with TypeScript
   - Schema validation with Zod

3. **API Security**
   - CORS configuration
   - Rate limiting
   - Input validation

## User Access

### Demo Credentials
```
Email: admin@example.com
Password: password123
```

### Role-Based Access
- Admin: Full access to all features
- User: Limited access to prediction features

## Development Workflow

1. **Local Development**
   ```bash
   # Install dependencies
   npm install

   # Set up environment variables
   cp .env.example .env

   # Initialize database
   npx prisma generate
   npx prisma db push
   npx prisma db seed

   # Start development server
   npm run dev
   ```

2. **Database Management**
   - Prisma migrations for schema changes
   - SQLite for development
   - Seeding script for demo data

## Testing

1. **Form Validation**
   - Zod schema validation
   - Real-time error feedback
   - Type safety checks

2. **API Integration**
   - Error handling
   - Loading states
   - Response validation

## Future Enhancements

1. **Planned Features**
   - Customer management dashboard
   - Analytics visualization
   - User settings and preferences
   - Batch prediction capabilities

2. **Technical Improvements**
   - API response caching
   - Performance optimization
   - Enhanced error handling
   - Comprehensive testing suite

## Monitoring and Maintenance

1. **Performance Monitoring**
   - API response times
   - Error rates
   - User engagement metrics

2. **Maintenance Tasks**
   - Regular dependency updates
   - Database backups
   - Security patches

## Conclusion

The CRM Admin Dashboard successfully implements a modern, secure, and user-friendly interface for customer churn prediction. The system demonstrates best practices in:
- Type safety and validation
- User experience and interface design
- Security and authentication
- API integration and error handling
- Database management and data persistence

The modular architecture allows for easy expansion and maintenance, while the comprehensive documentation ensures smooth onboarding for new developers. 