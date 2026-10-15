import { LoaderCircle, ShieldCheck } from "lucide-react";
import { useState, type FormEvent } from "react";

import { signIn } from "./auth";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(email.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Anmeldung fehlgeschlagen.");
      setBusy(false);
    }
  }

  return (
    <main className="login-page">
      <form className="login-card" onSubmit={submit}>
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={21} /></div>
          <div>
            <div className="brand-title">Kanzlei-Chatbot</div>
            <div className="brand-subtitle">Bitte melde dich an.</div>
          </div>
        </div>

        <label>
          E-Mail
          <input
            type="email"
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
            autoFocus
          />
        </label>
        <label>
          Passwort
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>

        {error && <div className="login-error">{error}</div>}

        <button type="submit" className="send-button" disabled={busy}>
          {busy && <LoaderCircle className="spin" size={15} />}
          Anmelden
        </button>
      </form>
    </main>
  );
}
