// A small dependency-free PDF writer for the downloadable assessment summary.
// It creates a text-only PDF locally; no assessment data leaves the browser.
const textForPdf = (value) => String(value ?? '')
  .replace(/[^\x20-\x7E]/g, '-')
  .replace(/\\/g, '\\\\')
  .replace(/\(/g, '\\(')
  .replace(/\)/g, '\\)');

function wrap(text, width = 78) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  words.forEach((word) => {
    const next = line ? `${line} ${word}` : word;
    if (next.length > width && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  });
  if (line) lines.push(line);
  return lines;
}

function pdfDocument(lines) {
  const content = [
    'BT',
    '/F1 19 Tf',
    '72 760 Td',
    '(MedFlowAI Prediction Summary) Tj',
    '/F1 10 Tf',
    '0 -28 Td',
    ...lines.flatMap((line) => [`(${textForPdf(line)}) Tj`, '0 -15 Td']),
    'ET',
  ].join('\n');
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
  ];
  let pdf = '%PDF-1.4\n%----\n';
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach((offset) => { pdf += `${String(offset).padStart(10, '0')} 00000 n \n`; });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
  return pdf;
}

export function downloadPredictionSummary({ user, heart, stroke, steps }) {
  const date = new Intl.DateTimeFormat(undefined, { dateStyle: 'long', timeStyle: 'short' }).format(new Date());
  const lines = [
    `Prepared for: ${user}`,
    `Generated: ${date}`,
    '',
    `Heart disease risk: ${heart.riskPercent} (${heart.tier})`,
    'Key factors:',
    ...heart.drivers.map((driver) => `  - ${driver}`),
    '',
    `Stroke risk: ${stroke.riskPercent} (${stroke.tier})`,
    'Key factors:',
    ...stroke.drivers.map((driver) => `  - ${driver}`),
    '',
    'Suggested next steps:',
    ...steps.flatMap((step, index) => wrap(`${index + 1}. ${step}`)),
    '',
    'Important: This prototype is not a diagnosis. Discuss results with a qualified clinician.',
  ];
  const blob = new Blob([pdfDocument(lines)], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `medflowai-prediction-summary-${new Date().toISOString().slice(0, 10)}.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
