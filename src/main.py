"""
Valura AI Microservice Entry Point.
"""
import logging
from textwrap import dedent

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.api.routes import router

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Valura AI Microservice",
    description="Intelligent routing and specialist agents for wealth management",
    version="1.0.0",
)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def home():
    html = dedent("""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Valura AI Microservice</title>
            <style>
                @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap");

                :root {
                    --bg: #0d1117;
                    --panel: #111827;
                    --accent: #30bced;
                    --accent-strong: #1f8ead;
                    --text: #e5e7eb;
                    --muted: #9ca3af;
                    --border: rgba(148, 163, 184, 0.18);
                }

                * {
                    box-sizing: border-box;
                }

                body {
                    margin: 0;
                    font-family: "Space Grotesk", "Segoe UI", sans-serif;
                    color: var(--text);
                    background: radial-gradient(1200px 600px at 20% -10%, #20314a 0%, transparent 65%),
                        radial-gradient(900px 500px at 90% 10%, #243a2f 0%, transparent 60%),
                        var(--bg);
                    min-height: 100vh;
                }

                .shell {
                    max-width: 980px;
                    margin: 0 auto;
                    padding: 48px 20px 64px;
                    animation: fadeIn 0.6s ease-out;
                }

                .hero {
                    display: grid;
                    gap: 12px;
                    margin-bottom: 32px;
                }

                .hero h1 {
                    font-size: clamp(28px, 4vw, 40px);
                    margin: 0;
                    letter-spacing: 0.2px;
                }

                .hero p {
                    margin: 0;
                    color: var(--muted);
                    font-size: 16px;
                }

                .pill-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }

                .pill {
                    padding: 6px 12px;
                    border-radius: 999px;
                    border: 1px solid var(--border);
                    color: var(--muted);
                    font-size: 12px;
                    background: rgba(17, 24, 39, 0.7);
                }

                .grid {
                    display: grid;
                    gap: 18px;
                }

                .card {
                    background: rgba(17, 24, 39, 0.9);
                    border: 1px solid var(--border);
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 12px 30px rgba(3, 7, 18, 0.35);
                    animation: floatIn 0.7s ease-out;
                }

                .card h2 {
                    margin: 0 0 12px;
                    font-size: 18px;
                }

                label {
                    display: block;
                    font-size: 13px;
                    color: var(--muted);
                    margin-bottom: 6px;
                }

                input,
                textarea {
                    width: 100%;
                    padding: 10px 12px;
                    border-radius: 10px;
                    border: 1px solid var(--border);
                    background: #0b1220;
                    color: var(--text);
                    font-family: inherit;
                    font-size: 14px;
                }

                textarea {
                    min-height: 100px;
                    resize: vertical;
                }

                .row {
                    display: grid;
                    gap: 14px;
                }

                .row.cols {
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                }

                .actions {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin-top: 12px;
                }

                button {
                    border: none;
                    border-radius: 10px;
                    padding: 10px 18px;
                    font-family: inherit;
                    font-size: 14px;
                    cursor: pointer;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }

                .primary {
                    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
                    color: #0b1220;
                    font-weight: 600;
                }

                .ghost {
                    background: transparent;
                    color: var(--text);
                    border: 1px solid var(--border);
                }

                button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                    transform: none;
                }

                button:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 10px 20px rgba(48, 188, 237, 0.15);
                }

                .status {
                    color: var(--muted);
                    font-size: 13px;
                }

                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
                    font-size: 13px;
                    color: #d1d5db;
                }

                .links {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin-top: 12px;
                }

                .links a {
                    color: var(--accent);
                    text-decoration: none;
                    font-size: 13px;
                }

                .examples {
                    margin-top: 16px;
                    display: grid;
                    gap: 10px;
                }

                .examples-title {
                    font-size: 13px;
                    color: var(--muted);
                }

                .examples-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }

                .example-btn {
                    border: 1px solid var(--border);
                    background: rgba(17, 24, 39, 0.65);
                    color: var(--text);
                    font-size: 12px;
                    padding: 8px 12px;
                    border-radius: 999px;
                }

                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }

                @keyframes floatIn {
                    from { opacity: 0; transform: translateY(12px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
            <script>
                function applyExample(text) {
                    const queryInput = document.getElementById("query");
                    const statusEl = document.getElementById("status");
                    if (!queryInput) {
                        return;
                    }
                    queryInput.value = text || "";
                    queryInput.focus();
                    if (statusEl) {
                        statusEl.textContent = "Example loaded";
                    }
                }
            </script>
        </head>
        <body>
            <main class="shell">
                <header class="hero">
                    <h1>Valura AI Microservice</h1>
                    <p>Query the portfolio intelligence pipeline and review the streamed response.</p>
                    <div class="pill-row">
                        <span class="pill">POST /api/v1/query</span>
                        <span class="pill">SSE streaming</span>
                        <span class="pill">FastAPI</span>
                    </div>
                </header>

                <div class="grid">
                    <section class="card">
                        <h2>Run a query</h2>
                        <form id="query-form" class="row" onsubmit="return streamQuery(event)">
                            <div class="row cols">
                                <div>
                                    <label for="userId">User ID</label>
                                    <input id="userId" name="userId" value="user_001_active_trader_us" />
                                </div>
                                <div>
                                    <label for="sessionId">Session ID (optional)</label>
                                    <input id="sessionId" name="sessionId" placeholder="session_123" />
                                </div>
                            </div>
                            <div>
                                <label for="query">Query</label>
                                <textarea id="query" name="query" placeholder="Give me a quick risk summary and any big concentration issues."></textarea>
                            </div>
                            <div class="actions">
                                <button class="primary" type="button" id="runBtn" onclick="streamQuery(event)">Stream response</button>
                                <button class="ghost" type="button" id="sampleBtn" onclick="applyExample('Summarize my portfolio risk and highlight any major concentration risks.')">Use sample query</button>
                            </div>
                            <div class="status" id="status">Idle</div>
                            <div class="examples">
                                <div class="examples-title">Starter examples</div>
                                <div class="examples-list">
                                    <button class="example-btn" type="button" data-example="Summarize my portfolio risk and highlight any major concentration risks." onclick="applyExample(this.dataset.example)">Risk + concentration</button>
                                    <button class="example-btn" type="button" data-example="Which holdings drive the most volatility? Provide a short breakdown." onclick="applyExample(this.dataset.example)">Volatility drivers</button>
                                    <button class="example-btn" type="button" data-example="Am I overexposed to a single sector or region?" onclick="applyExample(this.dataset.example)">Exposure check</button>
                                    <button class="example-btn" type="button" data-example="Suggest two diversification ideas based on my current holdings." onclick="applyExample(this.dataset.example)">Diversification ideas</button>
                                </div>
                            </div>
                        </form>
                    </section>

                    <section class="card">
                        <h2>Stream output</h2>
                        <pre id="output">Waiting for input...</pre>
                    </section>
                </div>

                <div class="links">
                    <a href="/docs" target="_blank" rel="noreferrer">OpenAPI Docs</a>
                    <a href="/health" target="_blank" rel="noreferrer">Health Check</a>
                </div>
            </main>

            <script>
                const output = document.getElementById("output");
                const statusEl = document.getElementById("status");
                const runBtn = document.getElementById("runBtn");

                const setStatus = (text) => {
                    statusEl.textContent = text;
                };

                async function streamQuery(event) {
                    if (event && typeof event.preventDefault === "function") {
                        event.preventDefault();
                    }
                    const userId = document.getElementById("userId").value.trim();
                    const sessionId = document.getElementById("sessionId").value.trim();
                    const query = document.getElementById("query").value.trim();

                    if (!query) {
                        setStatus("Enter a query to continue.");
                        return;
                    }

                    const payload = {
                        user_id: userId || "user_001_active_trader_us",
                        session_id: sessionId || null,
                        query,
                    };

                    output.textContent = "";
                    setStatus("Connecting...");
                    runBtn.disabled = true;

                    try {
                        const response = await fetch("/api/v1/query", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(payload),
                        });

                        if (!response.ok || !response.body) {
                            throw new Error("Request failed. Check server logs.");
                        }

                        setStatus("Streaming...");
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = "";

                        while (true) {
                            const { value, done } = await reader.read();
                            if (done) {
                                break;
                            }
                            buffer += decoder.decode(value, { stream: true });
                            buffer = buffer.replace(/\\r\\n/g, "\\n");
                            const parts = buffer.split("\\n\\n");
                            buffer = parts.pop() || "";

                            for (const part of parts) {
                                const lines = part.split("\\n");
                                let eventName = "";
                                const dataLines = [];

                                for (const line of lines) {
                                    if (line.startsWith("event:")) {
                                        eventName = line.slice(6).trim();
                                    }
                                    if (line.startsWith("data:")) {
                                        dataLines.push(line.slice(5).trim());
                                    }
                                }

                                if (dataLines.length) {
                                    const payloadText = dataLines.join("\\n");
                                    const prefix = eventName ? `[${eventName}] ` : "";
                                    output.textContent += `${prefix}${payloadText}\\n`;
                                }

                                if (eventName === "done") {
                                    setStatus("Done");
                                }
                            }
                        }

                        if (statusEl.textContent !== "Done") {
                            setStatus("Stream closed");
                        }
                    } catch (error) {
                        if (error.name === "AbortError") {
                            setStatus("Request canceled");
                        } else {
                            setStatus("Error: " + error.message);
                        }
                    } finally {
                        runBtn.disabled = false;
                    }
                    return false;
                }

                window.streamQuery = streamQuery;
            </script>
        </body>
        </html>
        """)
    return HTMLResponse(html)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
