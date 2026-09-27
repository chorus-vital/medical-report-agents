/**
 * The shape FastAPI's `_build_response` returns, and the one call that fetches it.
 *
 * These types mirror `src/api/routes/reports.py`. Everything Agent 3 produces has
 * already been checked in code against the lab rows, so the UI renders it as
 * given — but it is still model-authored text, so React's escaping is the floor,
 * never `dangerouslySetInnerHTML`.
 */

export type Flag = "GREEN" | "AMBER" | "RED" | "UNKNOWN";

export type EscalationLevel =
  | "no_data"
  | "routine"
  | "discuss_at_next_visit"
  | "see_doctor_promptly"
  | "seek_care_now";

export interface LabRow {
  test_name: string;
  standard_name: string | null;
  loinc_code: string | null;
  observed_value: string | number | null;
  unit: string | null;
  reference_range: string | null;
  flag: Flag;
  explanation: string | null;
  panel: string | null;
  match_confidence: number | null;
  range_source: string | null;
}

export interface EscalationReason {
  rule_id: string;
  test_name: string;
  detail: string;
}

export interface Analysis {
  report_id: string;
  filename: string;
  status: "success" | "degraded" | "no_results";
  patient_info: Record<string, string> | null;
  lab_results: LabRow[];
  report_notes: string[];
  extraction: { method: string; degraded: boolean; warnings: string[] };
  grounding: {
    loinc_coded_count: number;
    loinc_coded_of: number;
    flag_summary: Record<Flag, number>;
  };
  summary: string;
  key_findings: string[];
  doctor_questions: string[];
  lifestyle_tips: string[];
  reasoning: {
    escalation_level: EscalationLevel;
    escalation_reasons: EscalationReason[];
    degraded: boolean;
    confidence_score: number;
    dropped_claims: number;
  };
  errors: string[];
}

export const ESCALATION_LABEL: Record<EscalationLevel, string> = {
  no_data: "No results could be read",
  routine: "Nothing outside range",
  discuss_at_next_visit: "Raise at your next appointment",
  see_doctor_promptly: "Show a doctor soon",
  seek_care_now: "Contact a doctor today",
};

export async function analyseReport(file: File): Promise<Analysis> {
  const body = new FormData();
  body.append("file", file);

  const res = await fetch("/api/reports/analyze", { method: "POST", body });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(
      `The server could not read that file (${res.status}). ${detail.slice(0, 220)}`
    );
  }
  return res.json();
}

/** Pull the numeric bounds out of a printed range like "4000 - 10000". */
export function parseRange(text: string | null): [number, number] | null {
  if (!text) return null;
  const nums = text.replace(/,/g, "").match(/-?\d+(?:\.\d+)?/g);
  if (!nums || nums.length < 2) return null;
  const a = Number(nums[0]);
  const b = Number(nums[1]);
  if (Number.isNaN(a) || Number.isNaN(b)) return null;
  return [Math.min(a, b), Math.max(a, b)];
}

export function toNumber(value: string | number | null): number | null {
  if (value === null || value === undefined) return null;
  const n = Number(String(value).replace(/,/g, "").replace(/^[<>≤≥]=?/, "").trim());
  return Number.isNaN(n) ? null : n;
}

/** The single result most worth putting on screen by itself. */
export function headlineRow(rows: LabRow[]): LabRow | null {
  const red = rows.filter((r) => r.flag === "RED");
  const pool = red.length ? red : rows.filter((r) => r.flag === "AMBER");
  if (!pool.length) return null;

  // Furthest outside its interval, proportionally — so a wide range does not
  // automatically win over a narrow one.
  let best = pool[0];
  let bestScore = -Infinity;
  for (const row of pool) {
    const range = parseRange(row.reference_range);
    const value = toNumber(row.observed_value);
    if (!range || value === null) continue;
    const [low, high] = range;
    const span = high - low || 1;
    const score = value < low ? (low - value) / span : (value - high) / span;
    if (score > bestScore) {
      bestScore = score;
      best = row;
    }
  }
  return best;
}
