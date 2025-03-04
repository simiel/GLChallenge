"use client";

import * as React from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useToast } from "../../../../components/ui/use-toast";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "../../../../components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../../../components/ui/select";

const predictionSchema = z.object({
  age: z.string().transform((val) => parseInt(val, 10)).refine((val) => val >= 18 && val <= 100, {
    message: "Age must be between 18 and 100",
  }),
  gender: z.enum(["Male", "Female"]),
  location: z.enum(["Rural", "Urban", "Suburban"]),
  subscription_length: z.string().transform((val) => parseInt(val, 10)).refine((val) => val > 0, {
    message: "Subscription length must be greater than 0",
  }),
  monthly_bill: z.string().transform((val) => parseFloat(val)).refine((val) => val > 0, {
    message: "Monthly bill must be greater than 0",
  }),
  total_usage_gb: z.string().transform((val) => parseInt(val, 10)).refine((val) => val >= 0, {
    message: "Total usage must be greater than or equal to 0",
  }),
});

type PredictionValues = z.infer<typeof predictionSchema>;

export default function PredictionsPage() {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = React.useState(false);
  const [prediction, setPrediction] = React.useState<number | null>(null);

  const form = useForm<PredictionValues>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      age: 1,
      gender: undefined,
      location: undefined,
      subscription_length: 1,
      monthly_bill: 1,
      total_usage_gb: 1,
    },
  });

  async function onSubmit(data: PredictionValues) {
    setIsLoading(true);
    setPrediction(null);

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_PREDICTION_API_URL!, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
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
          Predict customer churn probability based on their profile and usage data.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Customer Information</CardTitle>
            <CardDescription>
              Enter customer details to generate a churn prediction.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="age"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Age</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="Enter age" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="gender"
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
                          <SelectItem value="Male">Male</SelectItem>
                          <SelectItem value="Female">Female</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="location"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Location</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select location" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="Rural">Rural</SelectItem>
                          <SelectItem value="Urban">Urban</SelectItem>
                          <SelectItem value="Suburban">Suburban</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="subscription_length"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Subscription Length (months)</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="Enter months" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="monthly_bill"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Monthly Bill ($)</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" placeholder="Enter amount" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="total_usage_gb"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total Usage (GB)</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="Enter GB" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? "Generating Prediction..." : "Generate Prediction"}
                </Button>
              </form>
            </Form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Prediction Result</CardTitle>
            <CardDescription>
              The predicted probability of customer churn.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {prediction === null ? (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                Fill out the form to generate a prediction
              </div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-lg bg-secondary p-6">
                  <div className="text-center">
                    <div className="text-2xl font-semibold">
                      {(prediction * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      Churn Probability
                    </div>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  {prediction < 0.3 ? (
                    <p>This customer has a low risk of churning. Consider offering loyalty rewards to maintain satisfaction.</p>
                  ) : prediction < 0.7 ? (
                    <p>This customer has a moderate risk of churning. Consider proactive engagement and service improvements.</p>
                  ) : (
                    <p>This customer has a high risk of churning. Immediate intervention and personalized retention strategies are recommended.</p>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 