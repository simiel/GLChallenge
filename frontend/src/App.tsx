import { QueryClient, QueryClientProvider } from 'react-query'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import Dashboard from './components/Dashboard'
import Predictions from './components/Predictions'
import Analytics from './components/Analytics'
import { Toaster } from './components/ui/toaster'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background">
        <header className="border-b">
          <div className="container flex h-16 items-center px-4">
            <h1 className="text-2xl font-bold">Customer Churn Prediction</h1>
          </div>
        </header>
        
        <main className="container py-6">
          <Tabs defaultValue="dashboard" className="space-y-4">
            <TabsList>
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
              <TabsTrigger value="predictions">Predictions</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>
            
            <TabsContent value="dashboard" className="space-y-4">
              <Dashboard />
            </TabsContent>
            
            <TabsContent value="predictions" className="space-y-4">
              <Predictions />
            </TabsContent>
            
            <TabsContent value="analytics" className="space-y-4">
              <Analytics />
            </TabsContent>
          </Tabs>
        </main>
        
        <Toaster />
      </div>
    </QueryClientProvider>
  )
}

export default App 