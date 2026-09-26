/**
 * Everything after the story: the questions, the full result set, and the
 * disclaimer that has to be on screen whether or not anyone scrolled.
 */

import { useState } from "react";
import type { Analysis, Flag, LabRow } from "../lib/api";

/** Questions as a horizontal accordion — one open at a time, click or hover. */
function Questions({ questions }: { questions: string[] }) {
  const [open, setOpen] = useState(0);
  if (!questions.length) return null;

  return (
    <section className="section wrap">
      <div className="section-head">
        <h2>Worth asking your doctor</h2>
        <p>
          Written from what actually came back on this report. Take them with
          you, or read them off your phone in the room.
        </p>
      </div>

      <div className="accordion">
        {questions.map((q, i) => (
          <button
            type="button"
            className="slice"
            key={i}
            data-open={open === i}
            onMouseEnter={() => setOpen(i)}
            onFocus={() => setOpen(i)}
            onClick={() => setOpen(i)}
            aria-expanded={open === i}
          >
            <span className="slice-hint">{String(i + 1).padStart(2, "0")}</span>
            <span className="slice-index">{String(i + 1).padStart(2, "0")}</span>
            <span className="slice-text">{q}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

/**
 * Every row, in a dense grid. Spans are chosen so each row of six closes
 * exactly: an abnormal result takes three columns, a normal one takes two.
 */
function AllResults({ rows }: { rows: LabRow[] }) {
  const [filter, setFilter] = useState<"all" | "abnormal">("all");

  const shown =
    filter === "abnormal"
      ? rows.filter((r) => r.flag === "RED" || r.flag === "AMBER")
      : rows;

  // Wide cells for what matters, narrow for the rest — then pad the tail so the
  // final row never leaves a hole.
  const spanFor = (flag: Flag) =>
    flag === "RED" ? "cell-3" : flag === "AMBER" ? "cell-3" : "cell-2";

  const units = shown.reduce(
    (n, r) => n + (r.flag === "RED" || r.flag === "AMBER" ? 3 : 2),
    0
  );
  const remainder = units % 6;
  const filler = remainder === 0 ? 0 : (6 - remainder) / 2;

  return (
    <section className="section wrap">
      <div className="section-head">
        <h2>All {rows.length} results</h2>
        <p>
          Coded to LOINC where the name could be matched confidently. A result
          the matcher was unsure of is left uncoded rather than guessed.
        </p>
      </div>

      <div style={{ display: "flex", gap: "0.6rem", marginBottom: "1.6rem" }}>
        <button
          type="button"
          className={filter === "all" ? "btn btn-solid" : "btn btn-ghost"}
          style={{ padding: "0.6rem 1.2rem", fontSize: "0.82rem" }}
          onClick={() => setFilter("all")}
        >
          Everything
        </button>
        <button
          type="button"
          className={filter === "abnormal" ? "btn btn-solid" : "btn btn-ghost"}
          style={{ padding: "0.6rem 1.2rem", fontSize: "0.82rem" }}
          onClick={() => setFilter("abnormal")}
        >
          Outside range only
        </button>
      </div>

      <div className="bento">
        {shown.map((row, i) => (
          <article
            className={`cell ${spanFor(row.flag)}`}
            data-flag={row.flag}
            key={`${row.test_name}-${i}`}
          >
            <div>
              <span className={`tag tag-${row.flag}`}>{row.flag}</span>
              <p className="cell-name" style={{ marginTop: "0.7rem" }}>
                {row.standard_name ?? row.test_name}
              </p>
            </div>
            <div className="cell-foot">
              <span className="cell-value">
                {String(row.observed_value ?? "-")}
                {row.unit ? (
                  <span style={{ fontSize: "0.7rem", opacity: 0.6 }}> {row.unit}</span>
                ) : null}
              </span>
              <span className="cell-range">{row.reference_range ?? "no range"}</span>
            </div>
          </article>
        ))}

        {Array.from({ length: filler }).map((_, i) => (
          <div className="cell cell-2" key={`filler-${i}`} aria-hidden="true" />
        ))}
      </div>
    </section>
  );
}

export default function Tail({ data, onReset }: { data: Analysis; onReset: () => void }) {
  const confidence = Math.round(data.reasoning.confidence_score * 100);
  const dropped = data.reasoning.dropped_claims;

  return (
    <>
      <Questions questions={data.doctor_questions} />

      {data.lifestyle_tips.length > 0 && (
        <section className="section wrap">
          <div className="section-head">
            <h2>Before you go in</h2>
          </div>
          <div className="bento">
            {data.lifestyle_tips.map((tip, i) => (
              <article className="cell cell-3" key={i}>
                <p className="cell-name" style={{ fontSize: "1rem", lineHeight: 1.55 }}>
                  {tip}
                </p>
              </article>
            ))}
            {data.lifestyle_tips.length % 2 === 1 && (
              <div className="cell cell-3" aria-hidden="true" />
            )}
          </div>
        </section>
      )}

      {data.report_notes.length > 0 && (
        <section className="section wrap">
          <div className="section-head">
            <h2>Printed on the report by the lab</h2>
            <p>Reproduced word for word, not paraphrased.</p>
          </div>
          {data.report_notes.map((note, i) => (
            <blockquote
              key={i}
              style={{
                margin: 0,
                borderLeft: "2px solid var(--line)",
                paddingLeft: "1.4rem",
                fontSize: "clamp(1.05rem, 2vw, 1.35rem)",
                lineHeight: 1.6,
                color: "var(--text)",
              }}
            >
              {note}
            </blockquote>
          ))}
        </section>
      )}

      <AllResults rows={data.lab_results} />

      <footer className="footer">
        <div className="wrap" style={{ padding: 0 }}>
          <p className="disclaimer">
            This is an automated reading of a lab report, not a medical
            diagnosis, and it does not name a cause for anything you see above.
            Every number here was checked against the report it came from.
            Decisions about what any of it means belong with a qualified
            clinician.
          </p>

          <div className="conf">
            <span>Confidence {confidence}%</span>
            <span className="conf-bar">
              <span className="conf-fill" style={{ width: `${confidence}%` }} />
            </span>
            {data.reasoning.degraded && (
              <span>Written from a template — the language model was unavailable.</span>
            )}
            {dropped > 0 && (
              <span>
                {dropped} statement{dropped === 1 ? "" : "s"} could not be matched
                to a result and {dropped === 1 ? "was" : "were"} removed.
              </span>
            )}
            {data.extraction.degraded && <span>Extraction was degraded.</span>}
          </div>

          <button
            type="button"
            className="btn btn-ghost"
            style={{ marginTop: "2.4rem" }}
            onClick={onReset}
          >
            Read another report
          </button>
        </div>
      </footer>
    </>
  );
}
