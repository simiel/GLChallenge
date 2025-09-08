export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-background">
      <div className="max-w-xl w-full p-8 bg-card rounded shadow">
        <h1 className="text-4xl font-bold mb-4 text-center">Welcome to CRM Admin</h1>
        <p className="mb-6 text-lg text-muted-foreground text-center">
          Streamline your customer management, analyze churn risk, and gain insights—all in one place.
        </p>
        <div className="flex flex-col gap-4">
          <a href="/auth/login" className="bg-primary text-white px-6 py-3 rounded text-center font-semibold hover:bg-primary/90 transition">
            Login
          </a>
          <a href="/dashboard" className="text-primary underline text-center">
            Go to Dashboard
          </a>
        </div>
      </div>
    </main>
  );
}
