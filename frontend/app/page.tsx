import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <section className="panel stack">
        <div className="stack" style={{ gap: 8 }}>
          <h1 style={{ margin: 0 }}>Aetheris</h1>
          <p className="muted" style={{ margin: 0 }}>
            Project foundation for the future cognitive system.
          </p>
        </div>

        <nav className="nav" aria-label="Aetheris sections">
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/chat">Chat</Link>
          <Link href="/memory">Memory</Link>
          <Link href="/settings">Settings</Link>
          <Link href="/developer">Developer</Link>
        </nav>
      </section>
    </main>
  );
}