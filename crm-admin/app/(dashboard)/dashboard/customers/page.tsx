import { Metadata } from "next";
import { prisma } from "@/lib/prisma";
import { DataTable } from "@/components/customers/data-table";
import { columns } from "@/components/customers/columns";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Customers | CRM Admin",
  description: "Manage your customers",
};

export default async function CustomersPage() {
  const customers = await prisma.customer.findMany({
    include: {
      assignedTo: true,
      predictions: {
        orderBy: {
          createdAt: "desc",
        },
        take: 1,
      },
    },
  });

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Customers</h1>
          <p className="text-muted-foreground">
            Manage and monitor your customer base
          </p>
        </div>
        <Button asChild>
          <Link href="/dashboard/customers/new">Add Customer</Link>
        </Button>
      </div>
      <DataTable columns={columns} data={customers} />
    </div>
  );
} 