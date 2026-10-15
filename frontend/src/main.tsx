
import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import type { User } from "firebase/auth";
import App from "./App";
import { isDemoMode } from "./api";
import { onUserChange, signOutUser } from "./auth";
import Login from "./Login";
import "./styles.css";

// undefined: the SDK has not yet restored the previous session
function Root() {
  const [user, setUser] = useState<User | null | undefined>(undefined);
  useEffect(() => (isDemoMode() ? undefined : onUserChange(setUser)), []);

  if (isDemoMode()) return <App />;
  if (user === undefined) return null;
  if (user === null) return <Login />;
  return (
    <App
      key={user.uid}
      userEmail={user.email ?? ""}
      onSignOut={() => void signOutUser()}
    />
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
