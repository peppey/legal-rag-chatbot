import { initializeApp } from "firebase/app";
import {
  getAuth,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
  type Auth,
  type User
} from "firebase/auth";

// Created on first use, so the demo mode works without a Firebase config.
let cachedAuth: Auth | undefined;

function firebaseAuth(): Auth {
  cachedAuth ??= getAuth(
    initializeApp({
      apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
      authDomain: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.firebaseapp.com`,
      projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID
    })
  );
  return cachedAuth;
}

export function onUserChange(callback: (user: User | null) => void): () => void {
  return onAuthStateChanged(firebaseAuth(), callback);
}

export async function signIn(email: string, password: string): Promise<void> {
  try {
    await signInWithEmailAndPassword(firebaseAuth(), email, password);
  } catch {
    throw new Error("E-Mail oder Passwort ist falsch.");
  }
}

export function signOutUser(): Promise<void> {
  return signOut(firebaseAuth());
}

// The SDK renews the token shortly before it expires.
export async function authHeaders(): Promise<Record<string, string>> {
  const token = await firebaseAuth().currentUser?.getIdToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class SessionExpiredError extends Error {
  constructor() {
    super("Die Sitzung ist abgelaufen. Bitte melde dich neu an.");
  }
}

export async function authFetch(
  input: string,
  init: RequestInit = {}
): Promise<Response> {
  const response = await fetch(input, {
    ...init,
    headers: { ...init.headers, ...(await authHeaders()) }
  });
  if (response.status === 401) {
    await signOutUser();
    throw new SessionExpiredError();
  }
  return response;
}
