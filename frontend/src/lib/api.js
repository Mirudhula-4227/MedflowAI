/* api.js — talks to the C++ inference server.
     POST /predict          { features }                                   -> { heart_risk, stroke_risk }
     POST /counterfactual   { features, model, flip_field, flip_value }    -> { original_risk, new_risk, delta }

   If the server is unreachable the app keeps working on MOCK responses
   that match those shapes exactly, so the UI can be demoed before the
   backend is up. `source` on the result says which one you got. */

export const API_BASE =
  (typeof window !== 'undefined' && window.MEDFLOW_API_BASE) || 'http://localhost:8080';

export const REQUEST_TIMEOUT_MS = 6000;

// Ephemeral Anonymous Session ID: Decoupled from personal identity (Zero PII storage)
export const ANONYMOUS_SESSION_ID =
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : 'sess_' + Math.random().toString(36).slice(2, 11);

async function post(path, body) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(API_BASE + path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': ANONYMOUS_SESSION_ID,
      },
      body: JSON.stringify(body),
      signal: ctl.signal,
    });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Server sent a non-JSON reply (HTTP ${res.status}).`);
    }
    if (!res.ok) throw new Error(data.error || `Server returned HTTP ${res.status}.`);
    return data;
  } finally {
    clearTimeout(timer);
  }
}

/* ---------------- mock inference ---------------- */
/* A small logistic stand-in so sliders move sensibly without a backend.
   Not the real model — never ship this as a prediction. */
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function mockPredict(f) {
  const heartZ =
    -6.4 +
    0.045 * (f.age - 50) +
    0.55 * f.sex +
    0.42 * (3 - f.cp) +
    0.016 * (f.trestbps - 120) +
    0.006 * (f.chol - 200) +
    0.3 * f.fbs +
    0.25 * f.restecg -
    0.026 * (f.thalach - 150) +
    0.75 * f.exang +
    0.6 * f.oldpeak +
    0.4 * f.slope +
    0.55 * f.ca +
    0.35 * (f.thal === 3 ? 1.6 : f.thal === 2 ? 0.8 : 0);

  const strokeZ =
    -7.1 +
    0.072 * (f.age - 45) +
    0.85 * f.hypertension +
    0.9 * f.heart_disease +
    0.012 * (f.avg_glucose_level - 100) +
    0.035 * (f.bmi - 25) +
    0.3 * f.ever_married +
    0.6 * f['smoking_status_smokes'] +
    0.35 * f['smoking_status_formerly smoked'] +
    0.15 * f.Residence_type +
    0.2 * f['work_type_Self-employed'];

  return {
    heart_risk: Number(sigmoid(heartZ).toFixed(4)),
    stroke_risk: Number(sigmoid(strokeZ).toFixed(4)),
  };
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------------- public calls ---------------- */

export async function predict(features) {
  try {
    const data = await post('/predict', { features });
    if (typeof data.heart_risk !== 'number' || typeof data.stroke_risk !== 'number') {
      throw new Error('Reply was missing heart_risk or stroke_risk.');
    }
    return { ...data, source: 'live' };
  } catch (err) {
    await wait(700); // let the loading screen breathe
    return { ...mockPredict(features), source: 'mock', reason: err.message };
  }
}

export async function counterfactual({ features, model, flip_field, flip_value }) {
  const payload = { features, model, flip_field, flip_value: Number(flip_value) };
  try {
    const data = await post('/counterfactual', payload);
    if (typeof data.new_risk !== 'number') throw new Error('Reply was missing new_risk.');
    return { ...data, source: 'live' };
  } catch {
    const key = model === 'heart' ? 'heart_risk' : 'stroke_risk';
    const original_risk = mockPredict(features)[key];
    const new_risk = mockPredict({ ...features, [flip_field]: Number(flip_value) })[key];
    return {
      original_risk,
      new_risk,
      delta: Number((new_risk - original_risk).toFixed(4)),
      source: 'mock',
    };
  }
}
