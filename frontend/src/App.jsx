import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { encodeFeatures, assertValidFeatures } from './lib/encoder.js';
import { QUESTIONS, SECTIONS, SAMPLE_ANSWERS } from './lib/questions.js';
import { predict, counterfactual, API_BASE } from './lib/api.js';
import { tierFor, overallTier, pct, explain, nextSteps, LEVERS } from './lib/risk.js';
import { currentUser, loadAssessment, register, saveAssessment, signIn, signOut } from './lib/userStorage.js';
import { downloadPredictionSummary } from './lib/pdf.js';

const TONE = { neutral: 'var(--brand)', heart: 'var(--heart)', stroke: 'var(--stroke)' };

/* ============================ chrome ============================ */

function Mark() {
  return (
    <span className="wordmark">
      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
        <path
          d="M2 12h3.4l1.7-4.4L10 16l2.4-8 1.9 4h5.7"
          fill="none"
          stroke="var(--pulse)"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      MedFlowAI
    </span>
  );
}

function Shell({ children, source, user, onSignOut }) {
  const live = source === 'live';
  const initials = user
    ? user.split(' ').map((w) => w[0]).join('').toUpperCase().slice(0, 2)
    : '?';
  return (
    <div className="shell">
      <header className="topbar">
        <Mark />
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span className="badge" data-live={String(live)}>
            {source
              ? live
                ? 'Connected to inference server'
                : 'Demo data — server unreachable'
              : API_BASE}
          </span>
          {user && (
            <div className="topbar-user">
              <div className="avatar" title={user}>{initials}</div>
              <button className="btn btn-signout" onClick={onSignOut}>Sign out</button>
            </div>
          )}
        </div>
      </header>
      <main className="stage">{children}</main>
    </div>
  );
}

/* ============================ login ============================ */

export function Login({ onLogin }) {
  const [mode, setMode]         = useState('signin');
  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [busy, setBusy]         = useState(false);
  const [err, setErr]           = useState('');
  const [showPw, setShowPw]     = useState(false);

  const attempt = async (e) => {
    e.preventDefault();
    setErr('');
    if (mode === 'register' && !name.trim()) { setErr('Enter your name.'); return; }
    if (!email.trim()) { setErr('Enter your email address.'); return; }
    if (!password)     { setErr('Enter your password.'); return; }
    if (mode === 'register' && password.length < 8) { setErr('Use at least 8 characters for your password.'); return; }
    if (mode === 'register' && password !== confirmPassword) { setErr('Passwords do not match.'); return; }
    setBusy(true);
    try {
      const user = mode === 'register'
        ? await register({ name, email, password })
        : await signIn(email, password);
      onLogin(user);
    } catch (error) {
      setErr(error.message || 'Unable to sign in.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-shell">
      {/* left decorative panel */}
      <div className="login-panel">
        <svg
          className="login-panel-trace"
          viewBox="0 0 600 800"
          preserveAspectRatio="xMidYMid slice"
          aria-hidden="true"
        >
          <path d="M-20 500 H120 l20 -30 l28 80 l36 -180 l32 160 l24 -60 h60 l30 -40 l28 70 h80 l30 -50 l28 80 l36 -180 l30 150 h120 l28 -40 l24 70 h80" />
          <path
            d="M-20 300 H80 l16 -20 l20 55 l26 -130 l22 110 l18 -45 h44 l22 -30 l20 50 h55 l22 -35 l20 55 l26 -130 l22 110 h88 l20 -28 l18 50 h620"
            transform="translate(0,100)"
            opacity="0.5"
          />
        </svg>
        <blockquote>
          "Prevention is the best cure. Knowing your risk is where that starts."
          <cite>— MedFlowAI, CodeCortex Hackathon 2024</cite>
        </blockquote>
      </div>

      {/* right form */}
      <div className="login-form-side">
        <Mark />
        <h2>{mode === 'register' ? 'Create your account' : 'Welcome back'}</h2>
        <p className="sub">{mode === 'register' ? 'Your assessments will be saved to this account on this browser.' : 'Sign in to continue where you left off.'}</p>

        <form onSubmit={attempt} noValidate>
          <div className="login-fields">
            {err && (
              <div className="login-err" role="alert">
                {err}
              </div>
            )}

            {mode === 'register' && (
              <div className="field-wrap">
                <label htmlFor="lf-name">Name</label>
                <input id="lf-name" autoComplete="name" value={name} onChange={(e) => { setName(e.target.value); setErr(''); }} />
              </div>
            )}

            <div className="field-wrap">
              <label htmlFor="lf-email">Email</label>
              <input
                id="lf-email"
                type="email"
                autoComplete="email"
                placeholder="doctor@medflowai.com"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setErr(''); }}
                aria-invalid={!!err}
              />
            </div>

            <div className="field-wrap">
              <label htmlFor="lf-pw">Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  id="lf-pw"
                  type={showPw ? 'text' : 'password'}
                  autoComplete="current-password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setErr(''); }}
                  aria-invalid={!!err}
                  style={{ paddingRight: 46 }}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  style={{
                    position: 'absolute', right: 14, top: '50%',
                    transform: 'translateY(-50%)', background: 'none', border: 'none',
                    cursor: 'pointer', padding: 0, color: 'var(--shell-ink-dim)', fontSize: 13,
                  }}
                  aria-label={showPw ? 'Hide password' : 'Show password'}
                >
                  {showPw ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

            {mode === 'register' && (
              <div className="field-wrap">
                <label htmlFor="lf-confirm">Confirm password</label>
                <input id="lf-confirm" type="password" autoComplete="new-password" value={confirmPassword} onChange={(e) => { setConfirmPassword(e.target.value); setErr(''); }} />
              </div>
            )}

            <button type="submit" className="btn login-btn" disabled={busy}>
              {busy ? 'Saving…' : mode === 'register' ? 'Create account' : 'Sign in'}
            </button>
          </div>
        </form>

        <p className="login-footer">
          {mode === 'signin' ? <><span>Demo: </span><strong style={{ color: 'var(--shell-ink)' }}>demo@medflowai.com</strong> / <strong style={{ color: 'var(--shell-ink)' }}>demo</strong><br />New here? <a href="#" onClick={(e) => { e.preventDefault(); setMode('register'); setErr(''); }}>Create an account</a>.</> : <>Already have an account? <a href="#" onClick={(e) => { e.preventDefault(); setMode('signin'); setErr(''); }}>Sign in</a>.</>}
        </p>
      </div>
    </div>
  );
}

/* ============================ landing ============================ */

function Landing({ onStart, onSample }) {
  return (
    <div className="col">
      <svg className="hero-trace" viewBox="0 0 700 92" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0 62 H150 l14 -6 l10 26 l16 -68 l17 60 l12 -18 h34 l13 -12 l11 22 h46 l16 -8 l12 22 l18 -66 l16 58 l13 -16 h38 l14 -10 l10 20 h122 l15 -7 l11 24 l17 -62 l15 52 h55" />
      </svg>
      <h1>Twenty-one questions, one at a time, and a number you can act on.</h1>
      <p className="lede">
        MedFlowAI runs your answers through two neural networks — one trained on cardiac
        catheterisation records, one on stroke outcomes — and shows you which single change
        would move your risk the most.
      </p>
      <div className="actions">
        <button className="btn on-shell" onClick={onStart}>Start the questions</button>
        <button className="btn ghost on-shell" onClick={onSample}>Fill with a sample patient</button>
      </div>
      <div className="facts">
        <div className="fact"><b>96.1%</b><span>Heart model test accuracy</span></div>
        <div className="fact"><b>80%</b><span>Stroke recall, imbalance corrected</span></div>
        <div className="fact"><b>~2 min</b><span>Typical time to finish</span></div>
      </div>
      <p className="disclaimer">
        This is a hackathon prototype, not a diagnosis. The models were trained on public
        datasets of a few thousand patients each and know nothing about your history beyond
        what you type here. Take the result to a clinician rather than acting on it alone.
      </p>
    </div>
  );
}

/* ============================ question ============================ */

function ChoiceInput({ q, value, onPick }) {
  return (
    <div className="opts" role="group" aria-label={q.prompt}>
      {q.options.map((o, i) => (
        <button
          key={String(o.value)}
          type="button"
          className="opt"
          aria-pressed={value === o.value}
          onClick={() => onPick(o.value)}
        >
          <span className="opt-key">{i + 1}</span>
          <span>
            <span className="opt-label">{o.label}</span>
            {o.note && <span className="opt-note">{o.note}</span>}
          </span>
        </button>
      ))}
    </div>
  );
}

function NumberInput({ q, value, onChange, onEnter, inputRef }) {
  return (
    <div className="numwrap">
      <input
        ref={inputRef}
        type="number"
        inputMode="decimal"
        value={value ?? ''}
        min={q.min}
        max={q.max}
        step={q.step}
        placeholder={q.placeholder}
        aria-label={q.prompt}
        onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
        onKeyDown={(e) => e.key === 'Enter' && onEnter()}
      />
      {q.unit && <span>{q.unit}</span>}
    </div>
  );
}

function SelectInput({ q, value, onChange }) {
  return (
    <select
      className="field"
      value={value ?? ''}
      aria-label={q.prompt}
      onChange={(e) => onChange(e.target.value || undefined)}
    >
      <option value="" disabled>Choose one</option>
      {q.options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

function QuestionScreen({ index, answers, setAnswer, onNext, onBack }) {
  const q = QUESTIONS[index];
  const value = answers[q.id];
  const [touched, setTouched] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    setTouched(false);
    if (q.type === 'number' && inputRef.current) inputRef.current.focus();
  }, [index, q.type]);

  const error = useMemo(() => {
    if (value === undefined || value === '') return 'Answer this to carry on.';
    if (q.type === 'number') {
      const n = Number(value);
      if (!Number.isFinite(n)) return 'Enter a number.';
      if (n < q.min || n > q.max) return `Enter a value between ${q.min} and ${q.max}.`;
    }
    return null;
  }, [value, q]);

  const advance = useCallback(() => {
    if (error) { setTouched(true); return; }
    onNext();
  }, [error, onNext]);

  useEffect(() => {
    if (q.type !== 'choice') return;
    const onKey = (e) => {
      const n = Number(e.key);
      if (n >= 1 && n <= q.options.length) setAnswer(q.id, q.options[n - 1].value);
      else if (e.key === 'Enter' && value !== undefined) onNext();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [q, value, setAnswer, onNext]);

  const section = SECTIONS[q.section];
  const tone = TONE[section.tone];
  const progress = ((index + 1) / QUESTIONS.length) * 100;

  return (
    <div className="col">
      <div className="card">
        <div className="rail">
          <div className="rail-track">
            <div className="rail-fill" style={{ width: `${progress}%`, background: tone }} />
          </div>
          <span className="rail-meta">{index + 1} of {QUESTIONS.length}</span>
        </div>
        <span className="section-tag" style={{ color: tone }}>
          <i className="dot" />{section.label}
        </span>
        <h2 className="prompt">{q.prompt}</h2>
        {q.help && <p className="help">{q.help}</p>}
        {q.type === 'choice' && <ChoiceInput q={q} value={value} onPick={(v) => setAnswer(q.id, v)} />}
        {q.type === 'number' && <NumberInput q={q} value={value} inputRef={inputRef} onChange={(v) => setAnswer(q.id, v)} onEnter={advance} />}
        {q.type === 'select' && <SelectInput q={q} value={value} onChange={(v) => setAnswer(q.id, v)} />}
        {touched && error && <p className="field-error">{error}</p>}
        <div className="actions">
          <button className="btn" onClick={advance} disabled={!!error && touched}>
            {index === QUESTIONS.length - 1 ? 'See my result' : 'Next'}
          </button>
          <button className="btn ghost" onClick={onBack}>Back</button>
          {q.type === 'choice' && <span className="hint">Press 1–{q.options.length} to pick</span>}
        </div>
      </div>
    </div>
  );
}

/* ============================ loading ============================ */

function LoadingScreen() {
  const lines = [
    'Standardising your 27 features',
    'Running the cardiac network',
    'Running the stroke network',
    'Working out what moved the number',
  ];
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((v) => (v + 1) % lines.length), 900);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="col loading">
      <svg className="ekg" viewBox="0 0 220 60" aria-hidden="true">
        <path d="M0 34 H58 l8 -4 l6 16 l10 -42 l10 36 l7 -10 h18 l8 -6 l7 14 h88" />
      </svg>
      <h2>Reading your answers</h2>
      <p aria-live="polite">{lines[i]}</p>
    </div>
  );
}

/* ============================ results ============================ */

function ScoreCard({ name, risk, tier, drivers }) {
  return (
    <section className="score">
      <div className="score-top">
        <span className="score-name">{name}</span>
        <span className="score-tier" style={{ color: tier.color }}>{tier.label}</span>
      </div>
      <div className="score-num" style={{ color: tier.color }}>{pct(risk)}</div>
      <div className="meter">
        <i style={{ width: `${Math.max(risk * 100, 2)}%`, background: tier.color }} />
      </div>
      <ul className="drivers">
        {drivers.map((d) => <li key={d}>{d}</li>)}
      </ul>
    </section>
  );
}

function Results({ features, result, onExplore, onRestart, error, user }) {
  const ex = explain(features, result.heart_risk, result.stroke_risk);
  const steps = nextSteps(features, result.heart_risk, result.stroke_risk);
  const tier = overallTier(result.heart_risk, result.stroke_risk);
  const headline =
    tier.key === 'high'
      ? 'Both numbers are worth a conversation with a doctor soon.'
      : tier.key === 'moderate'
        ? 'Nothing urgent, but there is room to move these numbers down.'
        : 'Your answers put you in the low band on both models.';

  return (
    <div className="col wide">
      <div className="card">
        {error && <div className="err">{error}</div>}
        <div className="result-head">
          <span className="tier-mark" style={{ background: tier.color }}>{tier.label} risk</span>
          <h2>{headline}</h2>
        </div>
        <div className="scores">
          <ScoreCard name="Heart disease" risk={ex.heart.risk} tier={ex.heart.tier} drivers={ex.heart.drivers} />
          <ScoreCard name="Stroke" risk={ex.stroke.risk} tier={ex.stroke.tier} drivers={ex.stroke.drivers} />
        </div>
        <div className="steps">
          <h3>What to do next</h3>
          <ol>{steps.map((s) => <li key={s}>{s}</li>)}</ol>
        </div>
        <details className="payload">
          <summary>See the exact payload sent to /predict</summary>
          <pre>{JSON.stringify({ features }, null, 2)}</pre>
        </details>
        <div className="actions">
          <button className="btn" onClick={onExplore}>Try changing one thing</button>
          <button
            className="btn ghost"
            onClick={() => downloadPredictionSummary({
              user,
              heart: { riskPercent: pct(ex.heart.risk), tier: ex.heart.tier.label, drivers: ex.heart.drivers },
              stroke: { riskPercent: pct(ex.stroke.risk), tier: ex.stroke.tier.label, drivers: ex.stroke.drivers },
              steps,
            })}
          >
            Download PDF summary
          </button>
          <button className="btn ghost" onClick={onRestart}>Start over</button>
        </div>
        {result.source === 'mock' && (
          <p className="mock-note">
            These numbers came from the built-in demo model because {API_BASE} did not answer.
            Start the C++ server and reload to see real inference.
          </p>
        )}
      </div>
    </div>
  );
}

/* ============================ counterfactual ============================ */

function Counterfactual({ features, baseline, onBack }) {
  const [lever, setLever] = useState(LEVERS[0]);
  const [value, setValue] = useState(() => Number(features[LEVERS[0].field]) || LEVERS[0].min);
  const [cf, setCf] = useState(null);
  const [busy, setBusy] = useState(false);

  // Fire immediately when the screen loads or the lever changes so
  // "Your answer" is populated before the user drags anything.
  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    setCf(null);
    const initialVal = Number(features[lever.field]) || lever.min;
    setValue(initialVal);
    counterfactual({
      features,
      model: lever.model,
      flip_field: lever.field,
      flip_value: initialVal,
    }).then((res) => {
      if (!cancelled) { setCf(res); setBusy(false); }
    });
    return () => { cancelled = true; };
  }, [lever, features]);

  // Debounced call when the user actually drags the slider
  useEffect(() => {
    const startVal = Number(features[lever.field]) || lever.min;
    if (Math.abs(value - startVal) < 1e-9) return;
    let cancelled = false;
    setBusy(true);
    const t = setTimeout(async () => {
      const res = await counterfactual({
        features,
        model: lever.model,
        flip_field: lever.field,
        flip_value: value,
      });
      if (!cancelled) { setCf(res); setBusy(false); }
    }, 220);
    return () => { cancelled = true; clearTimeout(t); };
  }, [value]);

  const original = cf ? cf.original_risk : baseline[lever.model === 'heart' ? 'heart_risk' : 'stroke_risk'];
  const updated = cf ? cf.new_risk : original;
  const delta = updated - original;
  const startValue = Number(features[lever.field]);
  const moved = Math.abs(value - startValue) > 1e-9;

  const sentence = !moved
    ? `Drag the slider to see what happens to your ${lever.model} risk.`
    : delta < -0.005
      ? `Dropping ${lever.label.toLowerCase()} to ${value}${lever.unit ? ' ' + lever.unit : ''} takes your ${lever.model} risk down by ${(Math.abs(delta) * 100).toFixed(1)} points.`
      : delta > 0.005
        ? `Moving ${lever.label.toLowerCase()} to ${value}${lever.unit ? ' ' + lever.unit : ''} pushes your ${lever.model} risk up by ${(delta * 100).toFixed(1)} points.`
        : `Changing ${lever.label.toLowerCase()} on its own barely shifts this model.`;

  return (
    <div className="col wide">
      <div className="card">
        <h2 className="prompt">What if one thing were different?</h2>
        <p className="help">
          Pick a factor, move the slider, and the server re-runs the same model with that one
          field replaced. Everything else stays exactly as you answered it.
        </p>
        <div className="lever-pick" role="group" aria-label="Factor to change">
          {LEVERS.map((l) => (
            <button key={l.field} type="button" className="chip" aria-pressed={lever.field === l.field} onClick={() => setLever(l)}>
              {l.label}
            </button>
          ))}
        </div>
        <div className="slider-row">
          <span className="slider-val">{value}{lever.unit && <span>{lever.unit}</span>}</span>
          <span className="hint">you answered {startValue}{lever.unit ? ` ${lever.unit}` : ''}</span>
        </div>
        <input type="range" min={lever.min} max={lever.max} step={lever.step} value={value} aria-label={lever.label} onChange={(e) => setValue(Number(e.target.value))} />
        <div className="cf-compare">
          <div className="cf-side">
            <small>Your answer</small>
            <b style={{ color: tierFor(original).color }}>{pct(original)}</b>
          </div>
          <div className="cf-arrow" aria-hidden="true">→</div>
          <div className="cf-side">
            <small>With the change</small>
            <b style={{ color: tierFor(updated).color, opacity: busy ? 0.45 : 1 }}>{pct(updated)}</b>
          </div>
        </div>
        <p className="cf-delta" aria-live="polite">{sentence}</p>
        <div className="actions">
          <button className="btn ghost" onClick={onBack}>Back to my result</button>
        </div>
        {cf && cf.source === 'mock' && (
          <p className="mock-note">Demo model — /counterfactual on {API_BASE} did not answer.</p>
        )}
      </div>
    </div>
  );
}

/* ============================ app ============================ */

export function App() {
  const [user, setUser]         = useState(() => currentUser());
  const saved = user ? loadAssessment(user.id) : null;
  const [screen, setScreen]     = useState(() => saved?.screen || 'landing');
  const [index, setIndex]       = useState(() => saved?.index || 0);
  const [answers, setAnswers]   = useState(() => saved?.answers || {});
  const [result, setResult]     = useState(() => saved?.result || null);
  const [features, setFeatures] = useState(() => saved?.features || null);
  const [error, setError]       = useState(() => saved?.error || null);

  useEffect(() => {
    if (user) saveAssessment(user.id, { screen, index, answers, result, features, error });
  }, [user, screen, index, answers, result, features, error]);

  const handleLogin = (nextUser) => {
    const assessment = loadAssessment(nextUser.id);
    setUser(nextUser);
    setScreen(assessment?.screen || 'landing');
    setIndex(assessment?.index || 0);
    setAnswers(assessment?.answers || {});
    setResult(assessment?.result || null);
    setFeatures(assessment?.features || null);
    setError(assessment?.error || null);
  };
  const handleSignOut = () => {
    signOut();
    setUser(null);
    setAnswers({}); setIndex(0); setResult(null); setFeatures(null); setError(null);
    setScreen('landing');
  };

  // Gate everything behind login
  if (!user) return <Login onLogin={handleLogin} />;

  const setAnswer = (id, value) => setAnswers((prev) => ({ ...prev, [id]: value }));

  const submit = async (finalAnswers) => {
    setScreen('loading');
    setError(null);
    let payload;
    try { payload = assertValidFeatures(encodeFeatures(finalAnswers)); }
    catch (e) { setError(e.message); setScreen('form'); return; }
    setFeatures(payload);
    const res = await predict(payload);
    setResult(res);
    setError(res.source === 'mock' && res.reason
      ? `Could not reach the inference server (${res.reason}) — showing demo numbers.`
      : null);
    setScreen('results');
  };

  const next = () => { if (index === QUESTIONS.length - 1) submit(answers); else setIndex((i) => i + 1); };
  const back = () => { if (index === 0) setScreen('landing'); else setIndex((i) => i - 1); };
  const restart = () => { setAnswers({}); setIndex(0); setResult(null); setFeatures(null); setError(null); setScreen('landing'); };

  const shellProps = { user: user.name, onSignOut: handleSignOut };

  if (screen === 'landing')
    return <Shell {...shellProps}><Landing onStart={() => { setIndex(0); setScreen('form'); }} onSample={() => { setAnswers(SAMPLE_ANSWERS); submit(SAMPLE_ANSWERS); }} /></Shell>;

  if (screen === 'form')
    return <Shell {...shellProps}><QuestionScreen index={index} answers={answers} setAnswer={setAnswer} onNext={next} onBack={back} /></Shell>;

  if (screen === 'loading')
    return <Shell {...shellProps}><LoadingScreen /></Shell>;

  if (screen === 'results')
    return <Shell source={result?.source} {...shellProps}><Results features={features} result={result} error={error} user={user.name} onExplore={() => setScreen('cf')} onRestart={restart} /></Shell>;

  return <Shell source={result?.source} {...shellProps}><Counterfactual features={features} baseline={result} onBack={() => setScreen('results')} /></Shell>;
}
