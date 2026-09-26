import { useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";

import { analyseReport, type Analysis } from "./lib/api";
import Story from "./components/Story";
import Tail from "./components/Tail";

gsap.registerPlugin(useGSAP);

type Phase = "idle" | "working" | "done";

function Nav({ data }: { data: Analysis | null }) {
  return (
    <nav className="nav">
      <span className="nav-mark">
        <i className="pulse" />
        Report Reader
      </span>
      {data && (
        <span className="nav-meta">
          {data.grounding.loinc_coded_count}/{data.grounding.loinc_coded_of} coded
        </span>
      )}
      <button
        type="button"
        className="nav-action"
        onClick={() =>
          window.scrollTo({
            top: document.body.scrollHeight,
            behavior: "smooth",
          })
        }
      >
        {data ? "All results" : "How it works"}
      </button>
    </nav>
  );
}

function Upload({
  phase,
  error,
  onPick,
  onRun,
  file,
}: {
  phase: Phase;
  error: string | null;
  file: File | null;
  onPick: (f: File | null) => void;
  onRun: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  useGSAP(
    () => {
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduced) return;

      gsap
        .timeline({ defaults: { ease: "power3.out" } })
        .from(".js-up-h1", { y: 34, opacity: 0, duration: 0.85 })
        .from(".js-up-lede", { y: 22, opacity: 0, duration: 0.7 }, "-=0.5")
        .from(".js-up-drop", { y: 26, opacity: 0, duration: 0.7 }, "-=0.45");
    },
    { scope: root }
  );

  return (
    <div className="stage" ref={root}>
      <div>
        <h1 className="js-up-h1">
          Your blood report, read back to you in plain English
        </h1>

        <p className="lede js-up-lede">
          Upload the PDF your lab sent. Every value is matched to a standard
          code, checked against the range printed on your own report, and
          explained without jargon. No condition is ever named.
        </p>

        <div
          className={`drop js-up-drop${over ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            onPick(e.dataTransfer.files?.[0] ?? null);
          }}
        >
          <input
            ref={input}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.webp,.docx,.txt,.csv"
            style={{ display: "none" }}
            onChange={(e) => onPick(e.target.files?.[0] ?? null)}
          />

          <p style={{ color: "var(--text-dim)", fontSize: "0.98rem" }}>
            Drop a PDF here, or choose a file
          </p>

          {file && <p className="filename">{file.name}</p>}

          <div className="cta-row">
            <button
              type="button"
              className="btn btn-solid"
              disabled={phase === "working"}
              onClick={() => (file ? onRun() : input.current?.click())}
            >
              {phase === "working" ? (
                <>
                  <span className="spinner" />
                  Reading your report
                </>
              ) : file ? (
                "Read this report"
              ) : (
                "Choose a file"
              )}
            </button>

            {file && phase !== "working" && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => input.current?.click()}
              >
                Pick a different one
              </button>
            )}
          </div>
        </div>

        {error && <p className="err">{error}</p>}
      </div>
    </div>
  );
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!file) return;
    setPhase("working");
    setError(null);
    try {
      const result = await analyseReport(file);
      setData(result);
      setPhase("done");
      window.scrollTo({ top: 0, behavior: "auto" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  }

  function reset() {
    setData(null);
    setFile(null);
    setPhase("idle");
    setError(null);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  return (
    <main>
      <div className="ambient" />
      <div className="grain" />

      <div className="shell">
        <Nav data={data} />

        {phase === "done" && data ? (
          <>
            <Story data={data} />
            <Tail data={data} onReset={reset} />
          </>
        ) : (
          <Upload
            phase={phase}
            error={error}
            file={file}
            onPick={(f) => {
              setFile(f);
              setError(null);
            }}
            onRun={run}
          />
        )}
      </div>
    </main>
  );
}
