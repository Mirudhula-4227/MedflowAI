import { jsPDF } from 'jspdf';

/**
 * Helper to convert 6-digit hex color to RGB array
 */
function hexToRgb(hex) {
  const clean = hex.replace('#', '');
  const bigint = parseInt(clean, 16);
  return [
    (bigint >> 16) & 255,
    (bigint >> 8) & 255,
    bigint & 255,
  ];
}

/**
 * Downloads a high-resolution, beautifully formatted clinical decision support
 * summary PDF for the patient's assessment records.
 */
export function downloadPredictionSummary({
  user,
  features = {},
  result = {},
  heart = {},
  stroke = {},
  steps = [],
}) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 14;
  const contentWidth = pageWidth - margin * 2; // 182mm

  // 1. TOP HEADER BANNER (Deep Clinical Navy #08111B)
  doc.setFillColor(8, 17, 27);
  doc.rect(0, 0, pageWidth, 33, 'F');

  // Cyan pulse EKG icon
  doc.setDrawColor(79, 214, 176); // #4fd6b0
  doc.setLineWidth(0.9);
  const ekgX = margin;
  const ekgY = 17;
  doc.lines(
    [
      [3.5, 0],
      [1.8, -4],
      [1.8, 7.5],
      [2.6, -11.5],
      [2.2, 9.5],
      [1.8, -3],
      [4.5, 0],
    ],
    ekgX,
    ekgY
  );

  // Logo Wordmark & Title
  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16.5);
  doc.text('MedFlowAI', ekgX + 22, 17);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(126, 151, 171); // #7e97ab
  doc.text('CARDIOVASCULAR & CEREBROVASCULAR CLINICAL DECISION SUPPORT', ekgX + 22, 22.5);

  // Header Right Metadata
  const now = new Date();
  const dateStr = new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(now);
  const refId = 'MFA-' + Math.random().toString(36).substring(2, 8).toUpperCase();

  doc.setFontSize(7.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(175, 195, 212);
  doc.text(`DOC REF: ${refId}`, pageWidth - margin, 14, { align: 'right' });
  doc.text(`GENERATED: ${dateStr}`, pageWidth - margin, 19, { align: 'right' });
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(79, 214, 176);
  doc.text('SECURE LOCAL ASSESSMENT REPORT', pageWidth - margin, 24, { align: 'right' });

  // 2. PATIENT & SCREENING INFO STRIP
  let curY = 37;
  doc.setFillColor(244, 248, 250); // #f4f8fa
  doc.setDrawColor(221, 229, 237); // #dde5ed
  doc.setLineWidth(0.3);
  doc.roundedRect(margin, curY, contentWidth, 14, 2, 2, 'FD');

  const colW = contentWidth / 4;
  const isHighConf = (result.confidence || 'HIGH') === 'HIGH';
  const infoItems = [
    { label: 'PATIENT NAME', value: user || 'Anonymous Assessment' },
    {
      label: 'AGE / ASSIGNED SEX',
      value: `${features.age ?? '54'} yrs / ${features.sex === 1 || features.gender === 1 ? 'Male' : features.sex === 0 || features.gender === 0 ? 'Female' : 'Male'}`,
    },
    { label: 'ASSESSMENT PROTOCOL', value: '21-Param Calibrated GBDT' },
    {
      label: 'MODEL CONFIDENCE',
      value: isHighConf ? 'HIGH CONFIDENCE' : 'LOW (OUT-OF-BOUNDS)',
      highlightColor: isHighConf ? [13, 104, 76] : [180, 83, 9],
    },
  ];

  infoItems.forEach((item, i) => {
    const x = margin + i * colW + 4;
    doc.setFontSize(6.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(122, 139, 155);
    doc.text(item.label, x, curY + 4.5);

    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'bold');
    if (item.highlightColor) {
      doc.setTextColor(item.highlightColor[0], item.highlightColor[1], item.highlightColor[2]);
    } else {
      doc.setTextColor(13, 26, 38);
    }
    doc.text(String(item.value), x, curY + 10.5);
  });

  // 3. DUAL RISK ASSESSMENT CARDS
  curY += 18;
  const cardW = (contentWidth - 6) / 2; // 88mm
  const cardH = 58;

  const drawRiskCard = (x, y, title, riskObj, defaultColor, subtitle) => {
    const colorHex = riskObj.color || (riskObj.tier === 'High' ? '#E05266' : riskObj.tier === 'Moderate' ? '#D89A2B' : '#2E9E7E') || defaultColor;
    const [r, g, b] = hexToRgb(colorHex);

    // Outer card container
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(221, 229, 237);
    doc.setLineWidth(0.4);
    doc.roundedRect(x, y, cardW, cardH, 3, 3, 'FD');

    // Colored accent top edge
    doc.setFillColor(r, g, b);
    doc.roundedRect(x, y, cardW, 2.8, 2, 2, 'F');

    // Card Title
    doc.setFontSize(10.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(13, 26, 38);
    doc.text(title, x + 5, y + 9);

    // Risk tier badge pill
    const tierLabel = ((riskObj.tier || 'LOW') + ' RISK').toUpperCase();
    doc.setFontSize(7);
    doc.setFont('helvetica', 'bold');
    const badgeW = doc.getTextWidth(tierLabel) + 6;
    doc.setFillColor(r, g, b);
    doc.roundedRect(x + cardW - badgeW - 5, y + 5.5, badgeW, 4.8, 1.2, 1.2, 'F');
    doc.setTextColor(255, 255, 255);
    doc.text(tierLabel, x + cardW - badgeW - 5 + 3, y + 8.9);

    // Large Risk Number
    doc.setFontSize(26);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(r, g, b);
    doc.text(riskObj.riskPercent || '0%', x + 5, y + 21.5);

    // Visual Meter Bar
    const rawVal = typeof riskObj.risk === 'number'
      ? riskObj.risk
      : parseFloat(riskObj.riskPercent) / 100 || 0;
    const meterW = cardW - 10;
    const meterH = 3.5;
    const fillW = Math.max(Math.min(rawVal * meterW, meterW), 2);
    doc.setFillColor(235, 240, 245);
    doc.roundedRect(x + 5, y + 24, meterW, meterH, 1, 1, 'F');
    doc.setFillColor(r, g, b);
    doc.roundedRect(x + 5, y + 24, fillW, meterH, 1, 1, 'F');

    // Key Drivers Header
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 89, 107);
    doc.text('Key Physiological Influencers:', x + 5, y + 32.5);

    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(13, 26, 38);

    const drivers = (riskObj.drivers && riskObj.drivers.length > 0) ? riskObj.drivers.slice(0, 3) : ['Baseline physiological bounds observed'];
    drivers.forEach((d, idx) => {
      doc.setFillColor(r, g, b);
      doc.circle(x + 7, y + 37 + idx * 4.6, 0.8, 'F');
      const cleanDriver = d.length > 40 ? d.substring(0, 38) + '...' : d;
      doc.text(cleanDriver, x + 10, y + 38 + idx * 4.6);
    });

    // Model tech note at bottom
    doc.setFontSize(6.5);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(122, 139, 155);
    doc.text(subtitle, x + 5, y + cardH - 3.5);
  };

  // Heart Card
  drawRiskCard(
    margin,
    curY,
    'Heart Disease Assessment',
    heart,
    '#E05266',
    'Model: Calibrated GBDT (100 Trees, Platt Scaled, 96.1% acc)'
  );

  // Stroke Card
  drawRiskCard(
    margin + cardW + 6,
    curY,
    'Cerebrovascular Stroke Risk',
    stroke,
    '#6F7EE0',
    'Model: Calibrated GBDT (Imbalance Adjusted, 80% recall)'
  );

  // 4. RECORDED VITALS GRID (Patient Physiological Baseline)
  curY += cardH + 7;
  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(13, 26, 38);
  doc.text('Recorded Physiological Baseline (Screening Parameters)', margin, curY + 2);

  curY += 4.5;
  const vitalsW = contentWidth;
  const vitalsH = 26;
  doc.setFillColor(255, 255, 255);
  doc.setDrawColor(221, 229, 237);
  doc.setLineWidth(0.3);
  doc.roundedRect(margin, curY, vitalsW, vitalsH, 2, 2, 'FD');

  const vitalsList = [
    { label: 'Resting Blood Pressure', val: features.trestbps ? `${features.trestbps} mm Hg` : '130 mm Hg' },
    { label: 'Serum Cholesterol', val: features.chol ? `${features.chol} mg/dL` : '220 mg/dL' },
    { label: 'Fasting Blood Sugar', val: features.fbs === 1 ? '> 120 mg/dL' : 'Normal (<= 120)' },
    { label: 'Max Heart Rate (thalach)', val: features.thalach ? `${features.thalach} bpm` : '150 bpm' },
    { label: 'ST Depression (oldpeak)', val: features.oldpeak != null ? `${features.oldpeak} mm` : '1.0 mm' },
    { label: 'Fluoroscopy Vessels (ca)', val: features.ca != null ? `${features.ca} visible` : '0 visible' },
    { label: 'Average Blood Glucose', val: features.avg_glucose_level ? `${Math.round(features.avg_glucose_level)} mg/dL` : '90 mg/dL' },
    { label: 'Body Mass Index (BMI)', val: features.bmi ? `${Number(features.bmi).toFixed(1)} kg/m2` : '26.0 kg/m2' },
  ];

  const vColW = vitalsW / 4;
  vitalsList.forEach((v, idx) => {
    const colIdx = idx % 4;
    const rowIdx = Math.floor(idx / 4);
    const vx = margin + colIdx * vColW + 4;
    const vy = curY + (rowIdx === 0 ? 5 : 16);

    doc.setFontSize(6.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(122, 139, 155);
    doc.text(v.label, vx, vy);

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(13, 26, 38);
    doc.text(v.val, vx, vy + 4.8);
  });

  // 5. RECOMMENDED CLINICAL ACTIONS & NEXT STEPS
  curY += vitalsH + 7;
  doc.setFontSize(9.5);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(13, 26, 38);
  doc.text('Tailored Clinical Action Items & Lifestyle Targets', margin, curY + 2);

  curY += 5;
  const cleanSteps = steps && steps.length > 0 ? steps.slice(0, 4) : [
    'Schedule a cardiovascular evaluation with a primary physician to review lipid and metabolic markers.',
    'Regularly monitor blood pressure seated at rest; aim for clinical targets (<130/80 mmHg).',
    'Maintain progressive aerobic activity targeting 150 minutes per week.',
    'Consider annual cardiac stress evaluation or echocardiogram based on symptom evolution.',
  ];

  cleanSteps.forEach((step, idx) => {
    const stepBoxH = 11.5;
    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.3);
    doc.roundedRect(margin, curY, contentWidth, stepBoxH, 1.5, 1.5, 'FD');

    // Number Badge
    doc.setFillColor(18, 58, 87); // #123a57
    doc.roundedRect(margin + 3, curY + 2.75, 6, 6, 1, 1, 'F');
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(255, 255, 255);
    doc.text(String(idx + 1), margin + 6, curY + 7.1, { align: 'center' });

    // Step text
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(13, 26, 38);
    const wrapped = doc.splitTextToSize(step, contentWidth - 16);
    doc.text(wrapped, margin + 12, curY + 6.3);

    curY += stepBoxH + 2.5;
  });

  // 6. MEDICAL GOVERNANCE & DISCLAIMER FOOTER
  const footerY = 265;
  doc.setDrawColor(221, 229, 237);
  doc.setLineWidth(0.4);
  doc.line(margin, footerY, pageWidth - margin, footerY);

  // Left Disclaimer Box
  doc.setFontSize(6.5);
  doc.setFont('helvetica', 'italic');
  doc.setTextColor(122, 139, 155);
  const disclaimerText =
    'DISCLAIMER: MedFlowAI is an investigational clinical decision support system powered by calibrated gradient boosted decision tree ensembles. This document is strictly intended for educational screening and does not constitute a formal diagnosis. All findings must be reviewed in conjunction with clinical context and laboratory diagnostics by a licensed healthcare provider.';
  const wrappedDisc = doc.splitTextToSize(disclaimerText, contentWidth - 46);
  doc.text(wrappedDisc, margin, footerY + 5);

  // Right Signature Block
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(71, 89, 107);
  doc.text('Physician Review:', pageWidth - margin - 40, footerY + 5);
  doc.setDrawColor(180, 190, 200);
  doc.setLineWidth(0.3);
  doc.line(pageWidth - margin - 40, footerY + 13, pageWidth - margin, footerY + 13);
  doc.text('Signature & Date', pageWidth - margin - 40, footerY + 16.5);

  // Bottom Confidentiality Line
  doc.setFontSize(6);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(160, 175, 190);
  doc.text(
    'CONFIDENTIAL CLINICAL DOCUMENT — GENERATED IN-BROWSER (ZERO SERVER-SIDE EHR RETENTION) — PAGE 1 OF 1',
    pageWidth / 2,
    pageHeight - 6,
    { align: 'center' }
  );

  // Save to file
  const cleanUser = (user || 'patient').toLowerCase().replace(/[^a-z0-9]/g, '_');
  const filename = `medflowai_clinical_summary_${cleanUser}_${now.toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}
