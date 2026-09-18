/**
 * Client-Side Assessment History Manager (Privacy-Preserving & Zero-Server Storage)
 * 
 * In accordance with MedFlowAI's data minimization and privacy-by-design principles,
 * assessment records are stored exclusively in the user's browser localStorage.
 * Health vitals and risk scores are never persisted on disk or database by the inference backend.
 */

const STORAGE_PREFIX = 'medflow_assessment_history_';

function getKey(user) {
  const safeUser = (user || 'guest').toLowerCase().replace(/[^a-z0-9]/g, '_');
  return `${STORAGE_PREFIX}${safeUser}`;
}

export function getHistory(user) {
  try {
    const raw = localStorage.getItem(getKey(user));
    if (!raw) return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch (err) {
    console.error('Failed to load assessment history:', err);
    return [];
  }
}

export function saveAssessmentToHistory(user, { features, result, answers }) {
  try {
    const list = getHistory(user);
    const now = new Date();
    const entry = {
      id: `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      timestamp: now.toISOString(),
      formattedDate: now.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }),
      heart_risk: result.heart_risk,
      stroke_risk: result.stroke_risk,
      confidence: result.confidence || 'HIGH',
      ood_distance: result.ood_distance != null ? result.ood_distance : null,
      hallucination_check: result.hallucination_check || null,
      source: result.source || 'live',
      features,
      answers: answers || {},
    };

    // Prepend latest entry and cap to 50 assessments
    const updated = [entry, ...list].slice(0, 50);
    localStorage.setItem(getKey(user), JSON.stringify(updated));
    return updated;
  } catch (err) {
    console.error('Failed to save assessment to history:', err);
    return getHistory(user);
  }
}

export function deleteAssessmentFromHistory(user, id) {
  try {
    const list = getHistory(user);
    const updated = list.filter((item) => item.id !== id);
    localStorage.setItem(getKey(user), JSON.stringify(updated));
    return updated;
  } catch (err) {
    console.error('Failed to delete history item:', err);
    return getHistory(user);
  }
}

export function clearUserHistory(user) {
  try {
    localStorage.removeItem(getKey(user));
    return [];
  } catch (err) {
    console.error('Failed to clear history:', err);
    return [];
  }
}
