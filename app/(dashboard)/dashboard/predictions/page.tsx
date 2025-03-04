"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { Loader2 } from "lucide-react"
import React from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../../components/ui/card"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
  FormDescription,
} from "../../../components/ui/form"
import { Input } from "../../../components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select"
import { Button } from "../../../components/ui/button"
import { useToast } from "../../../hooks/use-toast"
import { cn } from "../../../lib/utils"

const predictionSchema = z.object({
  TransactionCount: z.coerce.number().min(0, "Transaction count must be positive"),
  AverageTransactionAmount: z.coerce.number().min(0, "Average transaction amount must be positive"),
  TotalTransactionAmount: z.coerce.number().min(0, "Total transaction amount must be positive"),
  TransactionAmountStd: z.coerce.number().min(0, "Transaction amount standard deviation must be positive"),
  Age: z.coerce.number().min(18, "Age must be at least 18"),
  AccountBalance: z.coerce.number().min(0, "Account balance must be positive"),
  DaysSinceLastTransaction: z.coerce.number().min(0, "Days since last transaction must be positive"),
  CustomerTenure: z.coerce.number().min(0, "Customer tenure must be positive"),
  TransactionsPerMonth: z.coerce.number().min(0, "Transactions per month must be positive"),
  Gender: z.enum(["M", "F"]),
  Location: z.string().min(1, "Location is required"),
})

type PredictionFormValues = z.infer<typeof predictionSchema>

const defaultValues: Partial<PredictionFormValues> = {
  TransactionCount: 0,
  AverageTransactionAmount: 0,
  TotalTransactionAmount: 0,
  TransactionAmountStd: 0,
  Age: 18,
  AccountBalance: 0,
  DaysSinceLastTransaction: 0,
  CustomerTenure: 0,
  TransactionsPerMonth: 0,
  Gender: "M",
  Location: "",
}

export default function PredictionsPage() {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = React.useState(false)
  
  const form = useForm<PredictionFormValues>({
    resolver: zodResolver(predictionSchema),
    defaultValues,
  })

  async function onSubmit(data: PredictionFormValues) {
    try {
      setIsLoading(true)
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          TransactionCount: Number(data.TransactionCount),
          AverageTransactionAmount: Number(data.AverageTransactionAmount),
          TotalTransactionAmount: Number(data.TotalTransactionAmount),
          TransactionAmountStd: Number(data.TransactionAmountStd),
          Age: Number(data.Age),
          AccountBalance: Number(data.AccountBalance),
          DaysSinceLastTransaction: Number(data.DaysSinceLastTransaction),
          CustomerTenure: Number(data.CustomerTenure),
          TransactionsPerMonth: Number(data.TransactionsPerMonth),
          Gender: data.Gender,
          Location: data.Location.toUpperCase(),
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction")
      }

      const result = await response.json()
      
      toast({
        title: "Prediction Result",
        description: `Churn Probability: ${(result.probability * 100).toFixed(2)}%`,
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get prediction. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto py-10">
      <Card>
        <CardHeader>
          <CardTitle>Customer Churn Prediction</CardTitle>
          <CardDescription>
            Enter customer information to predict the likelihood of churn.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                  control={form.control}
                  name="TransactionCount"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Transaction Count</FormLabel>
                      <FormControl>
                        <Input type="number" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormDescription>
                        Total number of transactions
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="AverageTransactionAmount"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Average Transaction Amount</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="TotalTransactionAmount"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total Transaction Amount</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="TransactionAmountStd"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Transaction Amount Std Dev</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="Age"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Age</FormLabel>
                      <FormControl>
                        <Input type="number" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="AccountBalance"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Account Balance</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="DaysSinceLastTransaction"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Days Since Last Transaction</FormLabel>
                      <FormControl>
                        <Input type="number" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="CustomerTenure"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Customer Tenure (Days)</FormLabel>
                      <FormControl>
                        <Input type="number" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="TransactionsPerMonth"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Transactions Per Month</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" {...field} onChange={e => field.onChange(e.target.valueAsNumber)} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="Gender"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Gender</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select gender" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="M">Male</SelectItem>
                          <SelectItem value="F">Female</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="Location"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Location</FormLabel>
                      <FormControl>
                        <Input {...field} onChange={e => field.onChange(e.target.value.toUpperCase())} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Getting Prediction...
                  </>
                ) : (
                  "Get Prediction"
                )}
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  )
} 