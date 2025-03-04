"use client";

import * as React from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../../../components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "../../../../components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../../../components/ui/select";
import { Input } from "../../../../components/ui/input";
import { Button } from "../../../../components/ui/button";
import { useToast } from "../../../../components/ui/use-toast";
import { Loader2 } from "lucide-react";
import { cn } from "../../../../components/lib/utils";

const predictionSchema = z.object({
  TransactionCount: z.coerce.number().min(0, "Transaction count must be greater than or equal to 0"),
  AverageTransactionAmount: z.coerce.number().min(0, "Average transaction amount must be greater than or equal to 0"),
  TotalTransactionAmount: z.coerce.number().min(0, "Total transaction amount must be greater than or equal to 0"),
  TransactionAmountStd: z.coerce.number().min(0, "Transaction amount standard deviation must be greater than or equal to 0"),
  Age: z.coerce.number().min(18, "Age must be at least 18").max(100, "Age must be at most 100"),
  AccountBalance: z.coerce.number().min(0, "Account balance must be greater than or equal to 0"),
  DaysSinceLastTransaction: z.coerce.number().min(0, "Days since last transaction must be greater than or equal to 0"),
  CustomerTenure: z.coerce.number().min(0, "Customer tenure must be greater than or equal to 0"),
  TransactionsPerMonth: z.coerce.number().min(0, "Transactions per month must be greater than or equal to 0"),
  Gender: z.enum(["F", "M"]),
  Location: z.string().min(1, "Location is required"),
});

type PredictionValues = z.infer<typeof predictionSchema>;

export default function PredictionsPage() {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = React.useState(false);
  const [prediction, setPrediction] = React.useState<number | null>(null);

  const form = useForm<PredictionValues>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      TransactionCount: 0,
      AverageTransactionAmount: 0,
      TotalTransactionAmount: 0,
      TransactionAmountStd: 0,
      Age: 30,
      AccountBalance: 0,
      DaysSinceLastTransaction: 0,
      CustomerTenure: 0,
      TransactionsPerMonth: 0,
      Gender: undefined,
      Location: "",
    },
  });

  async function onSubmit(data: PredictionValues) {
    setIsLoading(true);
    setPrediction(null);

    try {
      const response = await fetch("https://glchallenge.onrender.com/api/predict", {
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
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const result = await response.json();
      setPrediction(result.churn_probability);
      
      toast({
        title: "Prediction Complete",
        description: "Successfully generated churn prediction.",
      });
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        title: "Error",
        description: "Failed to generate prediction. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="container mx-auto py-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Churn Prediction</h1>
        <p className="text-muted-foreground mt-2">
          Predict customer churn probability based on their transaction history and profile data.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Transaction Information</CardTitle>
            <CardDescription>
              Enter customer transaction and profile details to generate a churn prediction.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="TransactionCount"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Transaction Count</FormLabel>
                        <FormControl>
                          <Input type="number" placeholder="0" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="AverageTransactionAmount"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Avg Transaction Amount</FormLabel>
                        <FormControl>
                          <Input type="number" step="0.01" placeholder="0.00" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="TotalTransactionAmount"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Total Transaction Amount</FormLabel>
                        <FormControl>
                          <Input type="number" step="0.01" placeholder="0.00" {...field} />
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
                        <FormLabel>Transaction Amount Std</FormLabel>
                        <FormControl>
                          <Input type="number" step="0.01" placeholder="0.00" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="Age"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Age</FormLabel>
                        <FormControl>
                          <Input type="number" placeholder="30" {...field} />
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
                          <Input type="number" step="0.01" placeholder="0.00" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="DaysSinceLastTransaction"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Days Since Last Transaction</FormLabel>
                        <FormControl>
                          <Input type="number" placeholder="0" {...field} />
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
                        <FormLabel>Customer Tenure (days)</FormLabel>
                        <FormControl>
                          <Input type="number" placeholder="0" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="TransactionsPerMonth"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Transactions Per Month</FormLabel>
                        <FormControl>
                          <Input type="number" step="0.01" placeholder="0.00" {...field} />
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
                </div>

                <FormField
                  control={form.control}
                  name="Location"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Location</FormLabel>
                      <FormControl>
                        <Input placeholder="Enter customer location" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Prediction...
                    </>
                  ) : (
                    "Generate Prediction"
                  )}
                </Button>
              </form>
            </Form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Prediction Result</CardTitle>
            <CardDescription>
              The predicted probability of customer churn based on the provided data.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {prediction === null ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                {isLoading ? (
                  <div className="flex flex-col items-center space-y-4">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    <p>Analyzing customer data...</p>
                  </div>
                ) : (
                  "Fill out the form to generate a prediction"
                )}
              </div>
            ) : (
              <div className="space-y-6">
                <div className="rounded-lg bg-secondary p-6">
                  <div className="text-center">
                    <div className="text-4xl font-bold">
                      {(prediction * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">
                      Churn Probability
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className={cn(
                    "rounded-lg p-4",
                    prediction < 0.3 ? "bg-green-50 text-green-900" :
                    prediction < 0.7 ? "bg-yellow-50 text-yellow-900" :
                    "bg-red-50 text-red-900"
                  )}>
                    <h3 className="font-semibold mb-2">Risk Assessment</h3>
                    {prediction < 0.3 ? (
                      <p>Low Risk - This customer shows strong loyalty indicators. Consider offering premium services or loyalty rewards to maintain satisfaction.</p>
                    ) : prediction < 0.7 ? (
                      <p>Moderate Risk - Some warning signs present. Recommend proactive engagement through personalized offers and service quality improvements.</p>
                    ) : (
                      <p>High Risk - Immediate attention required. Implement targeted retention strategies and consider direct outreach to address potential issues.</p>
                    )}
                  </div>
                  
                  <div className="text-sm space-y-2">
                    <h4 className="font-semibold">Recommended Actions:</h4>
                    <ul className="list-disc pl-4 space-y-1">
                      {prediction < 0.3 ? (
                        <>
                          <li>Send personalized thank-you message</li>
                          <li>Offer early access to new features</li>
                          <li>Consider for loyalty program upgrade</li>
                        </>
                      ) : prediction < 0.7 ? (
                        <>
                          <li>Schedule account review call</li>
                          <li>Provide special discount or promotion</li>
                          <li>Send satisfaction survey</li>
                        </>
                      ) : (
                        <>
                          <li>Immediate account manager contact</li>
                          <li>Develop custom retention package</li>
                          <li>Priority support status</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 