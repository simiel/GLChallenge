import React from 'react'
import { useQuery } from 'react-query'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useToast } from './ui/use-toast'

interface FeatureImportance {
  feature_importance: Record<string, number>
}

const fetchFeatureImportance = async (): Promise<FeatureImportance> => {
  const response = await fetch('http://localhost:8000/model/features')
  if (!response.ok) throw new Error('Failed to fetch feature importance')
  return response.json()
}

export default function Analytics() {
  const { toast } = useToast()
  
  const { data, isLoading } = useQuery<FeatureImportance>(
    'featureImportance',
    fetchFeatureImportance,
    {
      onError: () => {
        toast({
          title: 'Error',
          description: 'Failed to load feature importance',
          variant: 'destructive',
        })
      },
    }
  )
  
  if (isLoading) {
    return <div>Loading...</div>
  }
  
  const featureData = data ? Object.entries(data.feature_importance)
    .map(([feature, importance]) => ({
      feature,
      importance,
    }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10) : []
  
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Top 10 Most Important Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={150} />
                <Tooltip />
                <Bar dataKey="importance" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Model Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold">Key Findings</h3>
              <ul className="list-disc list-inside mt-2 space-y-2">
                <li>Contract type is the most significant predictor of churn</li>
                <li>Monthly charges show strong correlation with churn probability</li>
                <li>Tenure length inversely correlates with churn risk</li>
                <li>Payment method preferences indicate different churn patterns</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold">Recommendations</h3>
              <ul className="list-disc list-inside mt-2 space-y-2">
                <li>Focus on converting month-to-month contracts to longer terms</li>
                <li>Implement tiered pricing strategies based on tenure</li>
                <li>Develop targeted retention programs for high-risk customers</li>
                <li>Monitor payment method changes as early warning signals</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 