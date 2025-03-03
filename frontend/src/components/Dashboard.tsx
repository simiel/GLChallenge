import { useQuery } from 'react-query'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useToast } from './ui/use-toast'

interface DataSummary {
  total_customers: number
  churn_rate: number
  numeric_features: string[]
  categorical_features: string[]
  missing_values: Record<string, number>
}

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
}

const fetchDataSummary = async (): Promise<DataSummary> => {
  const response = await fetch('http://localhost:8000/data/summary')
  if (!response.ok) throw new Error('Failed to fetch data summary')
  return response.json()
}

const fetchModelMetrics = async (): Promise<ModelMetrics> => {
  const response = await fetch('http://localhost:8000/model/metrics')
  if (!response.ok) throw new Error('Failed to fetch model metrics')
  return response.json()
}

export default function Dashboard() {
  const { toast } = useToast()
  
  const { data: summary, isLoading: summaryLoading } = useQuery<DataSummary>(
    'dataSummary',
    fetchDataSummary,
    {
      onError: () => {
        toast({
          title: 'Error',
          description: 'Failed to load data summary',
          variant: 'destructive',
        })
      },
    }
  )
  
  const { data: metrics, isLoading: metricsLoading } = useQuery<ModelMetrics>(
    'modelMetrics',
    fetchModelMetrics,
    {
      onError: () => {
        toast({
          title: 'Error',
          description: 'Failed to load model metrics',
          variant: 'destructive',
        })
      },
    }
  )
  
  if (summaryLoading || metricsLoading) {
    return <div>Loading...</div>
  }
  
  const metricsData = metrics ? [
    { name: 'Accuracy', value: metrics.accuracy },
    { name: 'Precision', value: metrics.precision },
    { name: 'Recall', value: metrics.recall },
    { name: 'F1 Score', value: metrics.f1 },
  ] : []
  
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Customers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary?.total_customers}</div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Churn Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {(summary?.churn_rate * 100).toFixed(1)}%
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Numeric Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary?.numeric_features.length}</div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Categorical Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{summary?.categorical_features.length}</div>
        </CardContent>
      </Card>
      
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 