import { notFound } from "next/navigation";

interface CustomerPageProps {
  params: { id: string };
}

async function getCustomer(id: string) {
  // TODO: Replace with real data fetching logic
  // Example placeholder data
  if (id === "demo") {
    return {
      id: "demo",
      name: "Demo Customer",
      email: "demo@example.com",
      phone: "123-456-7890",
      assignedTo: "Agent Smith",
      churnRisk: 0.12,
    };
  }
  return null;
}

export default async function CustomerPage({ params }: CustomerPageProps) {
  const customer = await getCustomer(params.id);
  if (!customer) return notFound();

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Customer Details</h1>
      <div className="rounded bg-muted p-6">
        <div><b>Name:</b> {customer.name}</div>
        <div><b>Email:</b> {customer.email}</div>
        <div><b>Phone:</b> {customer.phone}</div>
        <div><b>Assigned To:</b> {customer.assignedTo}</div>
        <div><b>Churn Risk:</b> {(customer.churnRisk * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}
