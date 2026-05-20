// ── GPU info ──────────────────────────────────────────────────────────────────

export interface GpuInfo {
  detected:     boolean;
  name:         string | null;
  total_gb:     number | null;
  available_gb: number | null;
}

export async function fetchGpuInfo(): Promise<GpuInfo> {
  const res = await fetch("/api/gpu");
  if (!res.ok) return { detected: false, name: null, total_gb: null, available_gb: null };
  return res.json();
}

// ── Run inputs ────────────────────────────────────────────────────────────────

export interface RunInputs {
  files:     File[];
  goal:      string;
  language:  string;
  objective: string;
  gpu_vram_gb: number;
  task?:     string;
}

export async function startRun(inputs: RunInputs): Promise<string> {
  const fd = new FormData();
  for (const f of inputs.files) fd.append("files", f);
  fd.append("goal",        inputs.goal);
  fd.append("language",    inputs.language);
  fd.append("objective",   inputs.objective);
  fd.append("gpu_vram_gb", String(inputs.gpu_vram_gb));
  if (inputs.task) fd.append("task", inputs.task);

  const res = await fetch("/api/runs", { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /api/runs failed: ${res.status} ${text}`);
  }
  const data = await res.json();
  return data.run_id as string;
}

// ── SSE events ───────────────────────────────────────────────────────────────

export interface RunEvent {
  kind: "open" | "log" | "done" | "error" | "confirm";
  data: string;
}

export interface QASamplePair {
  q: string;
  a: string;
}

export interface ConfirmPayload {
  action:        string;
  current_pairs: number;
  target_pairs:  number;
  sample_pairs?: QASamplePair[];
}

export async function confirmAction(
  runId: string,
  decision: "approve" | "refuse",
): Promise<void> {
  const res = await fetch(`/api/runs/${runId}/confirm?decision=${decision}`, {
    method: "POST",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`confirm failed: ${res.status} ${text}`);
  }
}

export async function uploadAdditionalFiles(
  runId: string,
  files: File[],
): Promise<string[]> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const res = await fetch(`/api/runs/${runId}/additional-files`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`upload additional files failed: ${res.status} ${text}`);
  }
  const data = await res.json();
  return data.files as string[];
}

export function subscribeToRun(
  runId: string,
  onEvent: (e: RunEvent) => void,
  onRetrying?: (retrying: boolean) => void,
): () => void {
  let es: EventSource | null = null;
  let stopped = false;
  let retryDelay = 3000;   // start at 3 s, back off up to 30 s
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (stopped) return;
    onRetrying?.(false);
    es = new EventSource(`/api/runs/${runId}/events`);

    es.addEventListener("open", (e) =>
      onEvent({ kind: "open", data: (e as MessageEvent).data ?? "" })
    );
    es.addEventListener("log", (e) =>
      onEvent({ kind: "log", data: (e as MessageEvent).data })
    );
    es.addEventListener("confirm", (e) =>
      onEvent({ kind: "confirm", data: (e as MessageEvent).data })
    );
    es.addEventListener("done", (e) => {
      onEvent({ kind: "done", data: (e as MessageEvent).data });
      stopped = true;
      es?.close();
    });

    es.onerror = () => {
      es?.close();
      if (stopped) return;
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
    if (retryTimer) clearTimeout(retryTimer);
    es?.close();
  };
}

// ── Run result (fallback poll when SSE drops) ─────────────────────────────────

export async function fetchRunResult(runId: string): Promise<{
  status: string; error?: string; result?: unknown;
} | null> {
  const res = await fetch(`/api/runs/${runId}/result`);
  if (!res.ok) return null;
  return res.json();
}

// ── Dataset ──────────────────────────────────────────────────────────────────

export interface DatasetResponse {
  count:  number;
  format: "alpaca" | "sharegpt";
  pairs:  unknown[];
}

export async function fetchDataset(runId: string): Promise<DatasetResponse> {
  const res = await fetch(`/api/runs/${runId}/dataset`);
  if (!res.ok) throw new Error(`dataset fetch failed: ${res.status}`);
  return res.json();
}

export function datasetDownloadUrl(runId: string): string {
  return `/api/runs/${runId}/dataset/download`;
}

// ── Training report ──────────────────────────────────────────────────────────

export interface TrainingHparams {
  lora_rank?:                    number;
  lora_alpha?:                   number;
  lora_dropout?:                 number;
  learning_rate?:                number;
  num_train_epochs?:             number;
  per_device_train_batch_size?:  number;
  gradient_accumulation_steps?:  number;
  warmup_ratio?:                 number;
  weight_decay?:                 number;
  [key: string]: unknown;
}

export interface TrainingReport {
  job_id:           string;
  model_id:         string;
  peft_method:      string;
  peft_source:      string;
  peft_rationale:   string;
  hparam_source:    string;
  hparam_rationale: string;
  task:             string;
  n_pairs:          number;
  hparams:          TrainingHparams;
  result: {
    success:    boolean;
    output_dir: string;
    metrics:    Record<string, number>;
    error?:     string;
  };
}

export async function fetchTrainingReport(runId: string): Promise<TrainingReport | null> {
  const res = await fetch(`/api/runs/${runId}/training-report`);
  if (!res.ok) return null;
  return res.json();
}

// ── Evaluation ───────────────────────────────────────────────────────────────

export interface EvalMetrics {
  rouge1:                       number;
  rouge2:                       number;
  rougeL:                       number;
  bleu4:                        number;
  n_test:                       number;
  quality_summary:              string;
  quality_tier?:                string;
  threshold_rouge1_good?:       number;
  threshold_rouge1_acceptable?: number;
  threshold_rouge2_good?:       number;
  threshold_rougeL_good?:       number;
  threshold_bleu4_good?:        number;
  llm_judge_accuracy?:          number;
  llm_judge_completeness?:      number;
  llm_judge_n?:                 number;
}

export interface DecisionJournalEntry {
  étape:         string;
  décision:      string;
  justification: string;
  statut:        "ok" | "warn" | "err" | "info";
}

export interface EvalReport {
  job_id:        string;
  adapter_path:  string;
  base_model_id: string;
  metrics:       EvalMetrics;
  samples:       unknown[];
}

export async function triggerEval(runId: string): Promise<void> {
  const res = await fetch(`/api/runs/${runId}/eval`, { method: "POST" });
  if (!res.ok && res.status !== 202) {
    const text = await res.text();
    throw new Error(`eval trigger failed: ${res.status} ${text}`);
  }
}

export async function fetchEvalReport(runId: string): Promise<EvalReport | null> {
  const res = await fetch(`/api/runs/${runId}/eval`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`eval fetch failed: ${res.status}`);
  return res.json();
}

// ── Exports ──────────────────────────────────────────────────────────────────

export function reportDownloadUrl(runId: string): string {
  return `/api/runs/${runId}/report/download`;
}

export function modelDownloadUrl(runId: string): string {
  return `/api/runs/${runId}/model/download`;
}

// ── Loss history ─────────────────────────────────────────────────────────────

export interface LossPoint {
  step:        number;
  epoch?:      number;
  train_loss?: number;
  eval_loss?:  number;
}

export interface LossHistory {
  job_id: string;
  points: LossPoint[];
}

export async function fetchLossHistory(runId: string): Promise<LossHistory | null> {
  const res = await fetch(`/api/runs/${runId}/loss-history`);
  if (res.status === 404) return null;
  if (!res.ok) return null;
  return res.json();
}

// ── Run history ──────────────────────────────────────────────────────────────

export interface RunSummary {
  job_id:               string;
  status:               string;
  task?:                string;
  domain?:              string;
  model_id?:            string;
  peft_method?:         string;
  primary_metric?:      string;
  primary_metric_value?: number;
  quality_tier?:        string;
  n_pairs?:             number;
  created_at?:          number;
  summary?:             string;
}

export async function fetchRunHistory(): Promise<RunSummary[]> {
  const res = await fetch("/api/runs");
  if (!res.ok) return [];
  const data = await res.json();
  return data.runs ?? [];
}

// ── Training images ───────────────────────────────────────────────────────────

export interface TrainingImage {
  name: string;
  url:  string;
}

export async function fetchTrainingImages(runId: string): Promise<TrainingImage[]> {
  const res = await fetch(`/api/runs/${runId}/images`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.images ?? [];
}
