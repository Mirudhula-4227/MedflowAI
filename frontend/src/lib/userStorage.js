// Browser-only account and assessment storage for the prototype.
// Passwords are never written as plaintext. A server-backed application should
// replace this module with real authentication, sessions, and encrypted storage.
const ACCOUNTS_KEY = 'medflowai.accounts.v1';
const SESSION_KEY = 'medflowai.session.v1';
const DATA_PREFIX = 'medflowai.assessment.v1.';

const DEMO_USERS = [
  { email: 'doctor@medflowai.com', password: 'heart2024', name: 'Dr. Ananya Krishnan' },
  { email: 'demo@medflowai.com', password: 'demo', name: 'Demo User' },
];

function read(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function write(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function accountFor({ email, name, passwordHash }) {
  return { id: `user-${crypto.randomUUID()}`, email, name, passwordHash, createdAt: new Date().toISOString() };
}

async function hashPassword(password) {
  const bytes = new TextEncoder().encode(password);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

const publicUser = ({ id, email, name }) => ({ id, email, name });

export function currentUser() {
  return read(SESSION_KEY, null);
}

export function signOut() {
  localStorage.removeItem(SESSION_KEY);
}

export async function signIn(email, password) {
  const normalizedEmail = email.trim().toLowerCase();
  const passwordHash = await hashPassword(password);
  const accounts = read(ACCOUNTS_KEY, []);
  let account = accounts.find((item) => item.email === normalizedEmail);

  // Keep the documented demo accounts available, while moving them into the
  // same persisted account store the first time they are used.
  if (!account) {
    const demo = DEMO_USERS.find((item) => item.email === normalizedEmail && item.password === password);
    if (demo) {
      account = accountFor({ email: normalizedEmail, name: demo.name, passwordHash });
      write(ACCOUNTS_KEY, [...accounts, account]);
    }
  }

  if (!account || account.passwordHash !== passwordHash) {
    throw new Error('Email or password is incorrect. Try demo@medflowai.com / demo.');
  }

  const user = publicUser(account);
  write(SESSION_KEY, user);
  return user;
}

export async function register({ name, email, password }) {
  const normalizedEmail = email.trim().toLowerCase();
  const accounts = read(ACCOUNTS_KEY, []);
  if (accounts.some((item) => item.email === normalizedEmail) || DEMO_USERS.some((item) => item.email === normalizedEmail)) {
    throw new Error('An account already exists for this email. Sign in instead.');
  }
  const account = accountFor({
    email: normalizedEmail,
    name: name.trim(),
    passwordHash: await hashPassword(password),
  });
  write(ACCOUNTS_KEY, [...accounts, account]);
  const user = publicUser(account);
  write(SESSION_KEY, user);
  return user;
}

export function loadAssessment(userId) {
  return read(DATA_PREFIX + userId, null);
}

export function saveAssessment(userId, assessment) {
  write(DATA_PREFIX + userId, { ...assessment, updatedAt: new Date().toISOString() });
}
