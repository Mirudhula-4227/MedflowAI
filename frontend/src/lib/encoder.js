/* ------------------------------------------------------------------
   encoder.js — THE ONLY place answers become model features.
   Nothing else in the app is allowed to build a feature object.

   Output: 27 keys. Heart model reads its 13 by name, stroke model
   reads its 15 by name (age is shared), the C++ server picks each
   model's slice via feature_names order. Every value is a Number.
   ------------------------------------------------------------------ */

/* Category -> one-hot flag name. "other" buckets map to null, which
   means every flag in that group stays 0. */
export const WORK_TYPE_FLAGS = {
  never_worked: 'work_type_Never_worked',
  private: 'work_type_Private',
  self_employed: 'work_type_Self-employed',
  children: 'work_type_children',
  govt_other: null, // Govt-or-other -> all four flags 0
};

export const SMOKING_FLAGS = {
  formerly: 'smoking_status_formerly smoked',
  never: 'smoking_status_never smoked',
  smokes: 'smoking_status_smokes',
  unknown: null, // Unknown -> all three flags 0
};

const ALL_ONE_HOT = [
  ...Object.values(WORK_TYPE_FLAGS),
  ...Object.values(SMOKING_FLAGS),
].filter(Boolean);

/* Sex is asked once and fed to both models: heart calls it `sex`,
   stroke calls it `gender`. Same 1 = male / 0 = female encoding.
   Set SHARE_SEX to false if you'd rather ask the two separately. */
export const SHARE_SEX = true;

const int = (v, fallback = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n) : fallback;
};

const num = (v, fallback = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

const bin = (v) => (v === true || v === 1 || v === '1' || v === 'yes' ? 1 : 0);

function oneHot(group, choice) {
  const out = {};
  for (const flag of Object.values(group)) if (flag) out[flag] = 0;
  const hit = group[choice];
  if (hit) out[hit] = 1;
  return out;
}

/**
 * @param {Object} a  raw answers keyed by question id
 * @returns {Object}  feature object ready for POST { features }
 */
export function encodeFeatures(a = {}) {
  const age = int(a.age, 0);
  const sex = bin(a.sex);

  const features = {
    /* ---- heart model: 13 ---- */
    age,
    sex,
    cp: int(a.cp),
    trestbps: int(a.trestbps),
    chol: int(a.chol),
    fbs: bin(a.fbs),
    restecg: int(a.restecg),
    thalach: int(a.thalach),
    exang: bin(a.exang),
    oldpeak: num(a.oldpeak), // float — ST depression
    slope: int(a.slope),
    ca: int(a.ca),
    thal: int(a.thal),

    /* ---- stroke model: 15 (age reused from above) ---- */
    gender: SHARE_SEX ? sex : bin(a.gender),
    hypertension: bin(a.hypertension),
    heart_disease: bin(a.heart_disease),
    ever_married: bin(a.ever_married),
    Residence_type: bin(a.Residence_type), // 1 = Urban, 0 = Rural
    avg_glucose_level: num(a.avg_glucose_level), // float
    bmi: num(a.bmi), // float
    ...oneHot(WORK_TYPE_FLAGS, a.work_type),
    ...oneHot(SMOKING_FLAGS, a.smoking_status),
  };

  return features;
}

/* Dev guard — throws loudly on the failure modes the backend can't
   recover from: strings, booleans, NaN, or a broken one-hot group. */
export function assertValidFeatures(f) {
  const bad = [];
  for (const [k, v] of Object.entries(f)) {
    if (typeof v !== 'number' || !Number.isFinite(v)) bad.push(`${k}=${JSON.stringify(v)}`);
  }
  for (const group of [WORK_TYPE_FLAGS, SMOKING_FLAGS]) {
    const flags = Object.values(group).filter(Boolean);
    const sum = flags.reduce((s, k) => s + (f[k] || 0), 0);
    if (sum > 1) bad.push(`one-hot group has ${sum} flags set: ${flags.join(', ')}`);
  }
  for (const k of ALL_ONE_HOT) {
    if (f[k] !== 0 && f[k] !== 1) bad.push(`${k} must be 0 or 1`);
  }
  if (bad.length) throw new Error('Invalid feature payload: ' + bad.join(' | '));
  return f;
}
