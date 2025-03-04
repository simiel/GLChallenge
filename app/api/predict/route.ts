import { NextResponse } from "next/server"
import { z } from "zod"

const predictionSchema = z.object({
  TransactionCount: z.number().min(0),
  AverageTransactionAmount: z.number().min(0),
  TotalTransactionAmount: z.number().min(0),
  TransactionAmountStd: z.number().min(0),
  Age: z.number().min(18),
  AccountBalance: z.number().min(0),
  DaysSinceLastTransaction: z.number().min(0),
  CustomerTenure: z.number().min(0),
  TransactionsPerMonth: z.number().min(0),
  Gender: z.enum(["M", "F"]),
  Location: z.string().min(1),
})

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const validatedData = predictionSchema.parse(body)

    // TODO: Replace with actual ML model prediction
    // For now, return a random probability
    const probability = Math.random()

    return NextResponse.json({ probability })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid request data", details: error.errors },
        { status: 400 }
      )
    }

    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
} 