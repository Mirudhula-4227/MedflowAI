/* questions.js — the flow, declared once.
   `id` matches the answer key the encoder reads. Nothing here writes
   feature names except the two dropdowns, which hand a category
   string to the encoder for one-hot expansion. */

export const SECTIONS = {
  profile: { label: 'About you', tone: 'neutral' },
  heart: { label: 'Heart', tone: 'heart' },
  stroke: { label: 'Stroke', tone: 'stroke' },
};

export const QUESTIONS = [
  {
    id: 'age',
    section: 'profile',
    prompt: 'How old are you?',
    help: 'Asked once. Both models use this same number.',
    type: 'number',
    unit: 'years',
    min: 1,
    max: 120,
    step: 1,
    placeholder: '54',
  },
  {
    id: 'sex',
    section: 'profile',
    prompt: 'What sex were you assigned at birth?',
    help: 'The heart and stroke models were both trained on this field.',
    type: 'choice',
    options: [
      { value: 1, label: 'Male' },
      { value: 0, label: 'Female' },
    ],
  },

  /* ---------------- heart ---------------- */
  {
    id: 'cp',
    section: 'heart',
    prompt: 'What kind of chest pain do you get?',
    type: 'choice',
    options: [
      { value: 0, label: 'Typical angina', note: 'Pressure on exertion, eases with rest' },
      { value: 1, label: 'Atypical angina', note: 'Chest pain that does not follow that pattern' },
      { value: 2, label: 'Non-anginal pain', note: 'Sharp or fleeting, unrelated to effort' },
      { value: 3, label: 'None at all', note: 'Asymptomatic' },
    ],
  },
  {
    id: 'trestbps',
    section: 'heart',
    prompt: 'What is your resting blood pressure?',
    help: 'The top number (systolic), taken sitting and rested.',
    type: 'number',
    unit: 'mm Hg',
    min: 80,
    max: 220,
    step: 1,
    placeholder: '130',
  },
  {
    id: 'chol',
    section: 'heart',
    prompt: 'What is your serum cholesterol?',
    help: 'Total cholesterol from your most recent blood panel.',
    type: 'number',
    unit: 'mg/dl',
    min: 100,
    max: 600,
    step: 1,
    placeholder: '220',
  },
  {
    id: 'fbs',
    section: 'heart',
    prompt: 'Is your fasting blood sugar above 120 mg/dl?',
    type: 'choice',
    options: [
      { value: 1, label: 'Yes' },
      { value: 0, label: 'No' },
    ],
  },
  {
    id: 'restecg',
    section: 'heart',
    prompt: 'What did your resting ECG show?',
    help: 'Copy this from the report if you have one.',
    type: 'choice',
    options: [
      { value: 0, label: 'Normal' },
      { value: 1, label: 'ST-T wave abnormality' },
      { value: 2, label: 'Left ventricular hypertrophy' },
    ],
  },
  {
    id: 'thalach',
    section: 'heart',
    prompt: 'What is the highest heart rate you reached during a stress test?',
    help: "If you have never had one, a rough estimate is 220 minus your age.",
    type: 'number',
    unit: 'bpm',
    min: 60,
    max: 220,
    step: 1,
    placeholder: '150',
  },
  {
    id: 'exang',
    section: 'heart',
    prompt: 'Does exercise bring on chest pain?',
    type: 'choice',
    options: [
      { value: 1, label: 'Yes' },
      { value: 0, label: 'No' },
    ],
  },
  {
    id: 'oldpeak',
    section: 'heart',
    prompt: 'How much ST depression did exercise cause?',
    help: 'From a stress ECG, measured in millimetres. Leave at 0 if unknown.',
    type: 'number',
    unit: 'mm',
    min: 0,
    max: 6.2,
    step: 0.1,
    placeholder: '1.0',
  },
  {
    id: 'slope',
    section: 'heart',
    prompt: 'Which way did the peak exercise ST segment slope?',
    type: 'choice',
    options: [
      { value: 0, label: 'Upsloping' },
      { value: 1, label: 'Flat' },
      { value: 2, label: 'Downsloping' },
    ],
  },
  {
    id: 'ca',
    section: 'heart',
    prompt: 'How many major vessels showed up on fluoroscopy?',
    type: 'choice',
    options: [0, 1, 2, 3, 4].map((v) => ({ value: v, label: String(v) })),
  },
  {
    id: 'thal',
    section: 'heart',
    prompt: 'What was your thalassemia scan result?',
    type: 'choice',
    options: [
      { value: 1, label: 'Normal' },
      { value: 2, label: 'Fixed defect' },
      { value: 3, label: 'Reversible defect' },
    ],
  },

  /* ---------------- stroke ---------------- */
  {
    id: 'hypertension',
    section: 'stroke',
    prompt: 'Have you been diagnosed with hypertension?',
    type: 'choice',
    options: [
      { value: 1, label: 'Yes' },
      { value: 0, label: 'No' },
    ],
  },
  {
    id: 'heart_disease',
    section: 'stroke',
    prompt: 'Have you been diagnosed with heart disease?',
    type: 'choice',
    options: [
      { value: 1, label: 'Yes' },
      { value: 0, label: 'No' },
    ],
  },
  {
    id: 'ever_married',
    section: 'stroke',
    prompt: 'Have you ever been married?',
    help: 'In the training data this stands in for age bracket and household support.',
    type: 'choice',
    options: [
      { value: 1, label: 'Yes' },
      { value: 0, label: 'No' },
    ],
  },
  {
    id: 'Residence_type',
    section: 'stroke',
    prompt: 'Where do you live?',
    type: 'choice',
    options: [
      { value: 1, label: 'Urban' },
      { value: 0, label: 'Rural' },
    ],
  },
  {
    id: 'avg_glucose_level',
    section: 'stroke',
    prompt: 'What is your average glucose level?',
    type: 'number',
    unit: 'mg/dl',
    min: 50,
    max: 300,
    step: 0.1,
    placeholder: '105.5',
  },
  {
    id: 'bmi',
    section: 'stroke',
    prompt: 'What is your BMI?',
    help: 'Weight in kilograms divided by height in metres, squared.',
    type: 'number',
    unit: '',
    min: 10,
    max: 70,
    step: 0.1,
    placeholder: '26.4',
  },
  {
    id: 'work_type',
    section: 'stroke',
    prompt: 'What best describes your work?',
    help: 'One-hot encoded into four flags. Government and anything else leaves all four at 0.',
    type: 'select',
    options: [
      { value: 'never_worked', label: 'Never worked' },
      { value: 'private', label: 'Private sector' },
      { value: 'self_employed', label: 'Self-employed' },
      { value: 'children', label: 'Child / in school' },
      { value: 'govt_other', label: 'Government or other' },
    ],
  },
  {
    id: 'smoking_status',
    section: 'stroke',
    prompt: 'Do you smoke?',
    help: 'One-hot encoded into three flags. Unknown leaves all three at 0.',
    type: 'select',
    options: [
      { value: 'formerly', label: 'Formerly smoked' },
      { value: 'never', label: 'Never smoked' },
      { value: 'smokes', label: 'Currently smoke' },
      { value: 'unknown', label: "Rather not say" },
    ],
  },
];

/* Demo values — used by "Fill with a sample patient" on the landing screen. */
export const SAMPLE_ANSWERS = {
  age: 61, sex: 1, cp: 0, trestbps: 148, chol: 244, fbs: 0, restecg: 1,
  thalach: 132, exang: 1, oldpeak: 1.8, slope: 1, ca: 1, thal: 3,
  hypertension: 1, heart_disease: 0, ever_married: 1, Residence_type: 1,
  avg_glucose_level: 168.4, bmi: 30.2, work_type: 'private', smoking_status: 'formerly',
};
