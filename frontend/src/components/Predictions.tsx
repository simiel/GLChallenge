import React, { useState } from 'react'
import { useMutation } from 'react-query'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { useToast } from './ui/use-toast'

interface PredictionResponse {
  churn_prediction: boolean
  churn_probability: number
}

interface CustomerFeatures {
  features: Record<string, number | string>
}

const predictChurn = async (data: CustomerFeatures): Promise<PredictionResponse> => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  if (!response.ok) throw new Error('Failed to make prediction')
  return response.json()
}

export default function Predictions() {
  const { toast } = useToast()
  const [features, setFeatures] = useState<Record<string, string>>({
    tenure: '',
    MonthlyCharges: '',
    TotalCharges: '',
    Contract_Month_to_month: '',
    Contract_One_year: '',
    Contract_Two_year: '',
    PaymentMethod_Credit_card: '',
    PaymentMethod_Electronic_check: '',
    PaymentMethod_Mailed_check: '',
    PaymentMethod_Bank_transfer: '',
  })

  const mutation = useMutation(predictChurn, {
    onSuccess: (data) => {
      toast({
        title: 'Prediction Complete',
        description: `Churn Probability: ${(data.churn_probability * 100).toFixed(1)}%`,
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to make prediction',
        variant: 'destructive',
      })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const numericFeatures = Object.entries(features).reduce((acc, [key, value]) => {
      acc[key] = key.includes('Contract_') || key.includes('PaymentMethod_')
        ? value === '1'
        : parseFloat(value)
      return acc
    }, {} as Record<string, number | boolean>)

    mutation.mutate({ features: numericFeatures })
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFeatures(prev => ({ ...prev, [name]: value }))
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Make Churn Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="tenure">Tenure (months)</Label>
              <Input
                id="tenure"
                name="tenure"
                type="number"
                value={features.tenure}
                onChange={handleChange}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="MonthlyCharges">Monthly Charges</Label>
              <Input
                id="MonthlyCharges"
                name="MonthlyCharges"
                type="number"
                value={features.MonthlyCharges}
                onChange={handleChange}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="TotalCharges">Total Charges</Label>
              <Input
                id="TotalCharges"
                name="TotalCharges"
                type="number"
                value={features.TotalCharges}
                onChange={handleChange}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="Contract_Month_to_month">Contract Type</Label>
              <Input
                id="Contract_Month_to_month"
                name="Contract_Month_to_month"
                type="number"
                value={features.Contract_Month_to_month}
                onChange={handleChange}
                placeholder="1 for Month-to-month, 0 otherwise"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="PaymentMethod_Credit_card">Payment Method</Label>
              <Input
                id="PaymentMethod_Credit_card"
                name="PaymentMethod_Credit_card"
                type="number"
                value={features.PaymentMethod_Credit_card}
                onChange={handleChange}
                placeholder="1 for Credit Card, 0 otherwise"
                required
              />
            </div>
          </div>
          
          <Button type="submit" disabled={mutation.isLoading}>
            {mutation.isLoading ? 'Predicting...' : 'Predict Churn'}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
} 