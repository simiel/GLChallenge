import { Env } from './types'

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url)
    
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    }
    
    // Handle OPTIONS request for CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: corsHeaders,
      })
    }
    
    try {
      // Route handling
      if (url.pathname === '/') {
        return new Response(JSON.stringify({ message: 'Welcome to the Customer Churn Prediction API' }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      if (url.pathname === '/health') {
        return new Response(JSON.stringify({ status: 'healthy' }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      if (url.pathname === '/predict' && request.method === 'POST') {
        const data = await request.json()
        
        // Call the Python backend service
        const response = await fetch(`${env.BACKEND_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data),
        })
        
        const result = await response.json()
        
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      if (url.pathname === '/model/features' && request.method === 'GET') {
        // Call the Python backend service
        const response = await fetch(`${env.BACKEND_URL}/model/features`)
        
        const result = await response.json()
        
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      if (url.pathname === '/data/summary' && request.method === 'GET') {
        // Call the Python backend service
        const response = await fetch(`${env.BACKEND_URL}/data/summary`)
        
        const result = await response.json()
        
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      if (url.pathname === '/model/metrics' && request.method === 'GET') {
        // Call the Python backend service
        const response = await fetch(`${env.BACKEND_URL}/model/metrics`)
        
        const result = await response.json()
        
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        })
      }
      
      // Handle 404
      return new Response(JSON.stringify({ error: 'Not Found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      })
      
    } catch (error) {
      return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      })
    }
  },
} 