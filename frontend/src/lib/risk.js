/* risk.js — turns probabilities and features into something a person
   can read. The server returns bare probabilities, so the wording,
   tiering and next steps are decided here. */

export const TIERS = {
  low: { key: 'low', label: 'Low', color: '#2E9E7E' },
  moderate: { key: 'moderate', label: 'Moderate', color: '#D89A2B' },
  high: { key: 'high', label: 'High', color: '#E05266' },
};

export function tierFor(p) {
  if (p < 0.33) return TIERS.low;
  if (p < 0.66) return TIERS.moderate;
  return TIERS.high;
}

export function overallTier(heart, stroke) {
  return tierFor(Math.max(heart, stroke));
}

export const pct = (p) => `${(p * 100).toFixed(1)}%`;

/* ---- what pushed the number up, in the user's words ---- */

const HEART_DRIVERS = [
  { test: (f) => f.ca >= 1, weight: 3, text: (f) => `${f.ca} major vessel${f.ca > 1 ? 's' : ''} showed narrowing on fluoroscopy` },
  { test: (f) => f.thal === 3, weight: 3, text: () => 'Your scan showed a reversible defect' },
  { test: (f) => f.exang === 1, weight: 3, text: () => 'Exercise brings on chest pain' },
  { test: (f) => f.oldpeak >= 1.5, weight: 2, text: (f) => `ST depression of ${f.oldpeak} mm during exertion` },
  { test: (f) => f.cp === 0, weight: 2, text: () => 'Your chest pain follows the typical angina pattern' },
  { test: (f) => f.chol >= 240, weight: 2, text: (f) => `Cholesterol of ${f.chol} mg/dl is above the 240 threshold` },
  { test: (f) => f.trestbps >= 140, weight: 2, text: (f) => `Resting blood pressure of ${f.trestbps} mm Hg is in the hypertensive range` },
  { test: (f) => f.thalach < 120, weight: 2, text: (f) => `Peak heart rate of ${f.thalach} bpm is lower than expected for your age` },
  { test: (f) => f.slope === 2, weight: 1, text: () => 'A downsloping ST segment at peak exercise' },
  { test: (f) => f.age >= 60, weight: 1, text: (f) => `Age ${f.age}` },
];

const STROKE_DRIVERS = [
  { test: (f) => f.hypertension === 1, weight: 3, text: () => 'A hypertension diagnosis' },
  { test: (f) => f.heart_disease === 1, weight: 3, text: () => 'An existing heart disease diagnosis' },
  { test: (f) => f.avg_glucose_level >= 140, weight: 3, text: (f) => `Average glucose of ${f.avg_glucose_level} mg/dl sits in the diabetic range` },
  { test: (f) => f['smoking_status_smokes'] === 1, weight: 3, text: () => 'Currently smoking' },
  { test: (f) => f.bmi >= 30, weight: 2, text: (f) => `BMI of ${f.bmi} is in the obese range` },
  { test: (f) => f['smoking_status_formerly smoked'] === 1, weight: 1, text: () => 'A history of smoking' },
  { test: (f) => f.age >= 60, weight: 2, text: (f) => `Age ${f.age}` },
];

function collect(list, f, limit = 3) {
  return list
    .filter((d) => d.test(f))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, limit)
    .map((d) => d.text(f));
}

export function explain(features, heartRisk, strokeRisk) {
  const heart = collect(HEART_DRIVERS, features);
  const stroke = collect(STROKE_DRIVERS, features);
  return {
    heart: {
      risk: heartRisk,
      tier: tierFor(heartRisk),
      drivers: heart.length ? heart : ['Nothing in your answers stood out as a cardiac risk factor.'],
    },
    stroke: {
      risk: strokeRisk,
      tier: tierFor(strokeRisk),
      drivers: stroke.length ? stroke : ['Nothing in your answers stood out as a stroke risk factor.'],
    },
  };
}

export function nextSteps(features, heartRisk, strokeRisk) {
  const top = Math.max(heartRisk, strokeRisk);
  const steps = [];

  if (top >= 0.66) {
    steps.push('Book an appointment with a doctor this week and bring these answers with you.');
  } else if (top >= 0.33) {
    steps.push('Raise this at your next routine check-up rather than waiting for symptoms.');
  } else {
    steps.push('Keep to your usual check-up schedule — nothing here calls for an urgent visit.');
  }

  if (features.trestbps >= 140 || features.hypertension === 1)
    steps.push('Track your blood pressure at home for two weeks and note the readings.');
  if (features.chol >= 240) steps.push('Ask for a repeat lipid panel and discuss whether statins fit your case.');
  if (features.avg_glucose_level >= 140) steps.push('Request an HbA1c test to confirm where your glucose sits over time.');
  if (features['smoking_status_smokes'] === 1)
    steps.push('Stopping smoking moves both numbers more than anything else on this list.');
  if (features.bmi >= 30) steps.push('A 5% drop in weight measurably shifts the stroke estimate.');
  if (features.exang === 1)
    steps.push('Chest pain during exercise is worth mentioning to a clinician directly, not just noting.');

  steps.push('Try the sliders on the next screen to see which single change moves your risk most.');
  return steps;
}

/* ---- levers offered on the counterfactual screen ----
   flip_field must be a real model feature name. */
export const LEVERS = [
  { field: 'chol', model: 'heart', label: 'Cholesterol', unit: 'mg/dl', min: 120, max: 400, step: 5 },
  { field: 'trestbps', model: 'heart', label: 'Resting blood pressure', unit: 'mm Hg', min: 90, max: 200, step: 2 },
  { field: 'thalach', model: 'heart', label: 'Peak heart rate', unit: 'bpm', min: 80, max: 202, step: 2 },
  { field: 'oldpeak', model: 'heart', label: 'ST depression', unit: 'mm', min: 0, max: 6.2, step: 0.1 },
  { field: 'avg_glucose_level', model: 'stroke', label: 'Average glucose', unit: 'mg/dl', min: 60, max: 280, step: 1 },
  { field: 'bmi', model: 'stroke', label: 'BMI', unit: '', min: 15, max: 55, step: 0.5 },
];
