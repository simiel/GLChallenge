'use client';

import { Card } from '@/components/ui/card';
import {
  Users,
  BarChart3,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';

const stats = [
  {
    name: 'Total Customers',
    value: '0',
    icon: Users,
    change: '+0%',
    changeType: 'increase',
  },
  {
    name: 'Active Customers',
    value: '0',
    icon: TrendingUp,
    change: '+0%',
    changeType: 'increase',
  },
  {
    name: 'Churn Risk',
    value: '0%',
    icon: AlertCircle,
    change: '-0%',
    changeType: 'decrease',
  },
  {
    name: 'Predictions Made',
    value: '0',
    icon: BarChart3,
    change: '+0%',
    changeType: 'increase',
  },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
        <p className="text-muted-foreground">
          Overview of your CRM system
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.name} className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <stat.icon className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    {stat.name}
                  </p>
                  <p className="text-2xl font-bold">{stat.value}</p>
                </div>
              </div>
              <div
                className={`text-sm ${
                  stat.changeType === 'increase'
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}
              >
                {stat.change}
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4 p-6">
          <h3 className="text-lg font-medium">Customer Churn Trends</h3>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            Chart will be implemented here
          </div>
        </Card>
        <Card className="col-span-3 p-6">
          <h3 className="text-lg font-medium">Recent Activity</h3>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            Activity feed will be implemented here
          </div>
        </Card>
      </div>
    </div>
  );
} 