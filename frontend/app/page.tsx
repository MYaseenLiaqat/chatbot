"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
type Message = { role: "user" | "assistant"; content: string };

async function request(path: string, options: RequestInit = {}) {
  const token = localStorage.getItem("profilepilot_session");
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API}${path}`, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "Request failed.");
  return body;
}

export default function Home() {
  const [token, setToken] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [github, setGithub] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState("");
  const [analysis, setAnalysis] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("profilepilot_session");
    if (saved) setToken(saved);
  }, []);

  async function ensureSession() {
    if (token) return token;
    const result = await request("/api/v1/sessions", { method: "POST" });
    localStorage.setItem("profilepilot_session", result.session_token);
    setToken(result.session_token);
    return result.session_token;
  }

  async function setup() {
    setBusy(true); setStatus("");
    try {
      await ensureSession();
      if (file) {
        const form = new FormData(); form.append("file", file);
        await request("/api/v1/documents/upload", { method: "POST", body: form });
      }
      if (github.trim()) {
        const result = await request("/api/v1/github/import", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: github })
        });
        setStatus(`Profile ready. Imported ${result.imported} repositories.`);
      } else setStatus("Profile ready. You can now chat with your CV.");
    } catch (error) { setStatus((error as Error).message); }
    finally { setBusy(false); }
  }

  async function sendMessage(text = message) {
    if (!text.trim()) return;
    setBusy(true); setMessages((items) => [...items, { role: "user", content: text }]); setMessage("");
    try {
      const result = await request("/api/v1/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      setMessages((items) => [...items, { role: "assistant", content: result.answer }]);
    } catch (error) { setStatus((error as Error).message); }
    finally { setBusy(false); }
  }

  async function analyzeJob() {
    setBusy(true); setAnalysis("");
    try {
      const result = await request("/api/v1/jobs/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: job })
      });
      setAnalysis(JSON.stringify(result, null, 2));
    } catch (error) { setAnalysis((error as Error).message); }
    finally { setBusy(false); }
  }

  async function reset() {
    if (token) await request("/api/v1/sessions", { method: "DELETE" }).catch(() => undefined);
    localStorage.removeItem("profilepilot_session"); setToken(""); setMessages([]); setStatus("Profile reset.");
  }

  return <main>
    <header><span className="brand">ProfilePilot AI</span><button className="secondary" onClick={reset}>Reset profile</button></header>
    <section className="hero">
      <div className="eyebrow">YOUR PROFESSIONAL PROFILE, AMPLIFIED</div>
      <h1>Your professional profile. Your personal AI assistant.</h1>
      <p className="subtitle">Upload your CV and connect your public GitHub projects to get grounded answers and practical job-fit preparation.</p>
    </section>
    <section className="workspace">
      <h2>Build your profile</h2>
      <label htmlFor="cv">CV (PDF, max 10 MB)</label>
      <input id="cv" type="file" accept=".pdf,application/pdf" onChange={(event) => setFile(event.target.files?.[0] || null)} />
      <label htmlFor="github">Public GitHub username (optional)</label>
      <input id="github" value={github} onChange={(event) => setGithub(event.target.value)} placeholder="octocat" />
      <div className="actions"><button onClick={setup} disabled={busy || (!file && !github)}>Build my AI profile</button></div>
      {status && <p className="status">{status}</p>}
    </section>
    <section className="card-grid">
      <article className="card"><h3>Grounded chat</h3><p>Answers are based on your uploaded evidence, with source references.</p></article>
      <article className="card"><h3>GitHub context</h3><p>Import public repositories and README content without OAuth or private access.</p></article>
      <article className="card"><h3>Job preparation</h3><p>Compare a job description with your actual skills and projects.</p></article>
    </section>
    <section className="workspace" style={{ marginTop: 24 }}>
      <h2>Ask your profile assistant</h2>
      <div className="chat">{messages.map((item, index) => <div key={index} className={`message ${item.role}`}>{item.content}</div>)}</div>
      <div className="actions">{["Summarize my profile", "What are my strongest technical skills?", "Which projects should I mention?"].map((item) => <button key={item} className="secondary" onClick={() => sendMessage(item)}>{item}</button>)}</div>
      <label htmlFor="message">Message</label>
      <textarea id="message" value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask about your CV or projects..." />
      <div className="actions"><button onClick={() => sendMessage()} disabled={busy || !message.trim()}>Send</button></div>
    </section>
    <section className="workspace" style={{ marginTop: 24 }}>
      <h2>Job description analyzer</h2>
      <textarea value={job} onChange={(event) => setJob(event.target.value)} placeholder="Paste a job description here..." />
      <div className="actions"><button onClick={analyzeJob} disabled={busy || job.length < 20}>Analyze fit</button></div>
      {analysis && <pre className="status">{analysis}</pre>}
    </section>
  </main>;
}
