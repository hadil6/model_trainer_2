// ── GPU info ──────────────────────────────────────────────────────────────────
export async function fetchGpuInfo() {
    const res = await fetch("/api/gpu");
    if (!res.ok)
        return { detected: false, name: null, total_gb: null, available_gb: null };
    return res.json();
}
export async function startRun(inputs) {
    const fd = new FormData();
    for (const f of inputs.files)
        fd.append("files", f);
    fd.append("goal", inputs.goal);
    fd.append("language", inputs.language);
    fd.append("objective", inputs.objective);
    fd.append("gpu_vram_gb", String(inputs.gpu_vram_gb));
    if (inputs.task)
        fd.append("task", inputs.task);
    const res = await fetch("/api/runs", { method: "POST", body: fd });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/runs failed: ${res.status} ${text}`);
    }
    const data = await res.json();
    return data.run_id;
}
export async function confirmAction(runId, decision) {
    const res = await fetch(`/api/runs/${runId}/confirm?decision=${decision}`, {
        method: "POST",
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`confirm failed: ${res.status} ${text}`);
    }
}
export async function uploadAdditionalFiles(runId, files) {
    const fd = new FormData();
    for (const f of files)
        fd.append("files", f);
    const res = await fetch(`/api/runs/${runId}/additional-files`, {
        method: "POST",
        body: fd,
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`upload additional files failed: ${res.status} ${text}`);
    }
    const data = await res.json();
    return data.files;
}
export function subscribeToRun(runId, onEvent, onRetrying) {
    let es = null;
    let stopped = false;
    let retryDelay = 3000; // start at 3 s, back off up to 30 s
    let retryTimer = null;
    function connect() {
        if (stopped)
            return;
        onRetrying?.(false);
        es = new EventSource(`/api/runs/${runId}/events`);
        es.addEventListener("open", (e) => onEvent({ kind: "open", data: e.data ?? "" }));
        es.addEventListener("log", (e) => onEvent({ kind: "log", data: e.data }));
        es.addEventListener("confirm", (e) => onEvent({ kind: "confirm", data: e.data }));
        es.addEventListener("done", (e) => {
            onEvent({ kind: "done", data: e.data });
            stopped = true;
            es?.close();
        });
        es.onerror = () => {
            es?.close();
            if (stopped)
                return;
            onRetrying?.(true);
            // Reconnect with exponential back-off (cap at 30 s)
            retryTimer = setTimeout(() => {
                retryDelay = Math.min(retryDelay * 1.5, 30_000);
                connect();
            }, retryDelay);
        };
    }
    connect();
    return () => {
        stopped = true;
        if (retryTimer)
            clearTimeout(retryTimer);
        es?.close();
    };
}
// ── Run result (fallback poll when SSE drops) ─────────────────────────────────
export async function fetchRunResult(runId) {
    const res = await fetch(`/api/runs/${runId}/result`);
    if (!res.ok)
        return null;
    return res.json();
}
export async function fetchDataset(runId) {
    const res = await fetch(`/api/runs/${runId}/dataset`);
    if (!res.ok)
        throw new Error(`dataset fetch failed: ${res.status}`);
    return res.json();
}
export function datasetDownloadUrl(runId) {
    return `/api/runs/${runId}/dataset/download`;
}
export async function fetchTrainingReport(runId) {
    const res = await fetch(`/api/runs/${runId}/training-report`);
    if (!res.ok)
        return null;
    return res.json();
}
export async function triggerEval(runId) {
    const res = await fetch(`/api/runs/${runId}/eval`, { method: "POST" });
    if (!res.ok && res.status !== 202) {
        const text = await res.text();
        throw new Error(`eval trigger failed: ${res.status} ${text}`);
    }
}
export async function fetchEvalReport(runId) {
    const res = await fetch(`/api/runs/${runId}/eval`);
    if (res.status === 404)
        return null;
    if (!res.ok)
        throw new Error(`eval fetch failed: ${res.status}`);
    return res.json();
}
// ── Exports ──────────────────────────────────────────────────────────────────
export function reportDownloadUrl(runId) {
    return `/api/runs/${runId}/report/download`;
}
export function modelDownloadUrl(runId) {
    return `/api/runs/${runId}/model/download`;
}
export async function fetchLossHistory(runId) {
    const res = await fetch(`/api/runs/${runId}/loss-history`);
    if (res.status === 404)
        return null;
    if (!res.ok)
        return null;
    return res.json();
}
export async function fetchRunHistory() {
    const res = await fetch("/api/runs");
    if (!res.ok)
        return [];
    const data = await res.json();
    return data.runs ?? [];
}
export async function fetchTrainingImages(runId) {
    const res = await fetch(`/api/runs/${runId}/images`);
    if (!res.ok)
        return [];
    const data = await res.json();
    return data.images ?? [];
}
