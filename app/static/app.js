/**
 * Agentic Fact-Checker Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const claimInput = document.getElementById("claimInput");
    const charCount = document.getElementById("charCount");
    const verifyForm = document.getElementById("verifyForm");
    const verifyBtn = document.getElementById("verifyBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const clearBtn = document.getElementById("clearBtn");
    const examplesContainer = document.getElementById("examplesContainer");
    
    const loadingPanel = document.getElementById("loadingPanel");
    const loadingStatusText = document.getElementById("loadingStatusText");
    const loadingSubText = document.getElementById("loadingSubText");
    
    const resultCard = document.getElementById("resultCard");
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTitle = document.getElementById("verdictTitle");
    const confidenceValue = document.getElementById("confidenceValue");
    const progressFill = document.getElementById("progressFill");
    const executionTimeValue = document.getElementById("executionTimeValue");
    const sourcesCountValue = document.getElementById("sourcesCountValue");
    const explanationText = document.getElementById("explanationText");
    const citationsWrapper = document.getElementById("citationsWrapper");
    const citationsList = document.getElementById("citationsList");
    const evidenceContainer = document.getElementById("evidenceContainer");
    
    const copyResultBtn = document.getElementById("copyResultBtn");
    const copyBtnText = document.getElementById("copyBtnText");
    const viewJsonBtn = document.getElementById("viewJsonBtn");
    const rawJsonPanel = document.getElementById("rawJsonPanel");
    const rawJsonCode = document.getElementById("rawJsonCode");
    
    const topKRetrieval = document.getElementById("topKRetrieval");
    const topKRerank = document.getElementById("topKRerank");

    // Stepper Elements
    const stepBadges = {
        search: document.getElementById("stepSearch"),
        retrieve: document.getElementById("stepRetrieve"),
        rerank: document.getElementById("stepRerank"),
        llm: document.getElementById("stepLLM"),
    };

    let latestResult = null;
    let stepAnimationInterval = null;

    // 1. Character Counter & Input Sync
    claimInput.addEventListener("input", () => {
        charCount.textContent = `${claimInput.value.length} / 1500`;
    });

    // Ctrl+Enter / Cmd+Enter to submit
    claimInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            verifyForm.dispatchEvent(new Event("submit"));
        }
    });

    // 2. Fetch Initial System Status & Example Claims
    loadSystemHealth();
    loadExamples();

    async function loadSystemHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                const statusDot = document.querySelector(".status-dot");
                const statusLabel = document.querySelector(".status-label");
                if (data.status === "operational") {
                    statusDot.style.backgroundColor = "#10b981";
                    statusLabel.textContent = `Online • ${data.llm_model}`;
                }
            }
        } catch (err) {
            console.warn("Health check skipped:", err);
        }
    }

    async function loadExamples() {
        try {
            const res = await fetch("/api/examples");
            if (res.ok) {
                const examples = await res.json();
                examplesContainer.innerHTML = "";
                examples.forEach((item) => {
                    const pill = document.createElement("button");
                    pill.type = "button";
                    pill.className = "example-pill";
                    pill.textContent = item.claim;
                    pill.addEventListener("click", () => {
                        claimInput.value = item.claim;
                        charCount.textContent = `${item.claim.length} / 1500`;
                        claimInput.focus();
                    });
                    examplesContainer.appendChild(pill);
                });
            }
        } catch (err) {
            console.warn("Could not load examples:", err);
        }
    }

    // 3. Clear Button
    clearBtn.addEventListener("click", () => {
        claimInput.value = "";
        charCount.textContent = "0 / 1500";
        resultCard.style.display = "none";
        loadingPanel.style.display = "none";
        rawJsonPanel.style.display = "none";
        resetPipelineSteps();
    });

    // 4. Form Submit Handler
    verifyForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const claim = claimInput.value.trim();
        if (!claim) return;

        // UI Loading State
        setLoadingState(true);
        startPipelineAnimation();

        try {
            const response = await fetch("/api/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    claim: claim,
                    top_k_retrieval: parseInt(topKRetrieval.value, 10) || 10,
                    top_k_rerank: parseInt(topKRerank.value, 10) || 5,
                }),
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            latestResult = data;
            renderResult(data);
        } catch (error) {
            alert(`Verification failed: ${error.message}`);
        } finally {
            setLoadingState(false);
            stopPipelineAnimation();
        }
    });

    function setLoadingState(isLoading) {
        verifyBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = "Verifying...";
            btnSpinner.style.display = "inline-block";
            loadingPanel.style.display = "block";
            resultCard.style.display = "none";
            loadingPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } else {
            btnText.textContent = "Verify Claim with AI";
            btnSpinner.style.display = "none";
            loadingPanel.style.display = "none";
        }
    }

    // Pipeline Stepper Progress Simulator
    function startPipelineAnimation() {
        const steps = [
            { id: "search", title: "Searching DuckDuckGo live sources...", sub: "Retrieving candidate web results" },
            { id: "retrieve", title: "Computing Dense Semantic Vectors...", sub: "Embedding queries with SentenceTransformers" },
            { id: "rerank", title: "Neural Cross-Encoder Reranking...", sub: "ms-marco-MiniLM precision evaluation" },
            { id: "llm", title: "Gemini Synthesis & Fact Verification...", sub: "Synthesizing evidence and computing confidence" },
        ];

        let currentIndex = 0;
        resetPipelineSteps();

        const advance = () => {
            const current = steps[currentIndex];
            if (!current) return;

            resetPipelineSteps();
            if (stepBadges[current.id]) {
                stepBadges[current.id].classList.add("active");
            }
            loadingStatusText.textContent = current.title;
            loadingSubText.textContent = current.sub;

            currentIndex = (currentIndex + 1) % steps.length;
        };

        advance();
        stepAnimationInterval = setInterval(advance, 1400);
    }

    function stopPipelineAnimation() {
        clearInterval(stepAnimationInterval);
        resetPipelineSteps();
    }

    function resetPipelineSteps() {
        Object.values(stepBadges).forEach((el) => el && el.classList.remove("active"));
    }

    // 5. Render Result
    function renderResult(data) {
        resultCard.style.display = "block";
        rawJsonPanel.style.display = "none";

        // Verdict Banner Configuration
        const verdict = (data.verdict || "UNCERTAIN").toUpperCase();
        verdictTitle.textContent = verdict;
        verdictBanner.className = "verdict-banner";

        if (verdict === "SUPPORTED") {
            verdictBanner.classList.add("supported");
            verdictIcon.textContent = "✓";
        } else if (verdict === "REFUTED") {
            verdictBanner.classList.add("refuted");
            verdictIcon.textContent = "✕";
        } else {
            verdictBanner.classList.add("uncertain");
            verdictIcon.textContent = "？";
        }

        // Metrics
        const conf = data.confidence || 0;
        confidenceValue.textContent = `${conf}%`;
        progressFill.style.width = `${Math.min(Math.max(conf, 0), 100)}%`;
        executionTimeValue.textContent = `${data.execution_time_seconds || 0}s`;
        sourcesCountValue.textContent = (data.evidence || []).length;

        // Explanation (Format paragraphs)
        explanationText.innerHTML = "";
        const paragraphs = (data.explanation || "No explanation provided.").split("\n\n");
        paragraphs.forEach((p) => {
            if (p.trim()) {
                const pEl = document.createElement("p");
                pEl.textContent = p.trim();
                explanationText.appendChild(pEl);
            }
        });

        // Citations
        citationsList.innerHTML = "";
        if (data.citations && data.citations.length > 0) {
            citationsWrapper.style.display = "block";
            data.citations.forEach((url) => {
                const a = document.createElement("a");
                a.className = "citation-item";
                a.href = url;
                a.target = "_blank";
                a.rel = "noopener noreferrer";
                a.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                        <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                    </svg>
                    <span>${url}</span>
                `;
                citationsList.appendChild(a);
            });
        } else {
            citationsWrapper.style.display = "none";
        }

        // Evidence Cards
        evidenceContainer.innerHTML = "";
        if (data.evidence && data.evidence.length > 0) {
            data.evidence.forEach((item, idx) => {
                const card = document.createElement("div");
                card.className = "evidence-card";
                card.innerHTML = `
                    <div class="evidence-top">
                        <span class="evidence-title">#${idx + 1} — ${escapeHtml(item.title || "Untitled")}</span>
                        <span class="score-badge">Score: ${item.score.toFixed(4)}</span>
                    </div>
                    <p class="evidence-snippet">${escapeHtml(item.content)}</p>
                    ${item.url && item.url !== "#" ? `
                        <a href="${item.url}" target="_blank" rel="noopener noreferrer" class="evidence-link">
                            <span>Open Source</span>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                <polyline points="15 3 21 3 21 9"></polyline>
                                <line x1="10" y1="14" x2="21" y2="3"></line>
                            </svg>
                        </a>
                    ` : ""}
                `;
                evidenceContainer.appendChild(card);
            });
        } else {
            evidenceContainer.innerHTML = "<p style='color: var(--text-muted); font-size: 0.9rem;'>No evidence was retrieved.</p>";
        }

        // Update Raw JSON snippet
        rawJsonCode.textContent = JSON.stringify(data, null, 2);

        // Smooth scroll to results
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // 6. Copy Result
    copyResultBtn.addEventListener("click", () => {
        if (!latestResult) return;
        const textToCopy = `CLAIM: ${latestResult.claim}\nVERDICT: ${latestResult.verdict} (${latestResult.confidence}% confidence)\n\nEXPLANATION:\n${latestResult.explanation}\n\nCITATIONS:\n${(latestResult.citations || []).join("\n")}`;
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtnText.textContent = "Copied!";
            setTimeout(() => { copyBtnText.textContent = "Copy Result"; }, 2000);
        });
    });

    // 7. Toggle Raw JSON
    viewJsonBtn.addEventListener("click", () => {
        const isHidden = rawJsonPanel.style.display === "none";
        rawJsonPanel.style.display = isHidden ? "block" : "none";
    });

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
