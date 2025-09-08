import { notFound } from "next/navigation";

interface EditCustomerPageProps {
  params: { id: string };
}

async function getCustomer(id: string) {
  // TODO: Replace with real data fetching logic
  if (id === "demo") {
    return {
      id: "demo",
      name: "Demo Customer",
      email: "demo@example.com",
      phone: "123-456-7890",
      assignedTo: "Agent Smith",
    };
  }
  return null;
}

export default async function EditCustomerPage({ params }: EditCustomerPageProps) {
  const customer = await getCustomer(params.id);
  if (!customer) return notFound();

  // Placeholder form (no real update logic)
  return (
    <div className="p-8 max-w-xl">
      <h1 className="text-2xl font-bold mb-4">Edit Customer</h1>
      <form className="space-y-4 bg-muted p-6 rounded">
        <div>
          <label className="block font-medium mb-1">Name</label>
          <input className="w-full p-2 rounded border" defaultValue={customer.name} />
        </div>
        <div>
          <label className="block font-medium mb-1">Email</label>
          <input className="w-full p-2 rounded border" defaultValue={customer.email} />
        </div>
        <div>
          <label className="block font-medium mb-1">Phone</label>
          <input className="w-full p-2 rounded border" defaultValue={customer.phone} />
        </div>
        <div>
          <label className="block font-medium mb-1">Assigned To</label>
          <input className="w-full p-2 rounded border" defaultValue={customer.assignedTo} />
        </div>
        <button type="submit" className="bg-primary text-white px-4 py-2 rounded">Save</button>
      </form>
    </div>
  );
}
