/**
 * Vercel Serverless: Proxy POST /api/ask to backend (Railway/Render/BACKEND_URL).
 * Set BACKEND_URL in Vercel Environment Variables to your FastAPI RAG backend.
 */
module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method Not Allowed' });
  }

  const backendUrl = process.env.BACKEND_URL || '';
  if (!backendUrl) {
    return res.status(503).json({
      detail: 'Backend yapılandırılmadı. Vercel → Settings → Environment Variables → BACKEND_URL ekleyin (örn. https://your-app.railway.app).',
    });
  }

  const url = backendUrl.replace(/\/$/, '') + '/ask';
  let body;
  try {
    body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  } catch {
    return res.status(400).json({ detail: 'Geçersiz JSON' });
  }

  if (!body || !body.question) {
    return res.status(400).json({ detail: 'question alanı gerekli' });
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: String(body.question).trim() }),
    });
    const data = await response.json().catch(() => ({}));
    res.status(response.status).json(data);
  } catch (err) {
    console.error('Backend proxy error:', err.message);
    res.status(502).json({
      detail: 'RAG backend\'e bağlanılamadı. BACKEND_URL doğru mu? ' + (err.message || ''),
    });
  }
};
