/**
 * The scroll-driven read of one report.
 *
 * Four pinned panels, each scrubbed to scroll position: the count, the 25
 * results sorting themselves, the single worst value against its interval, and
 * the summary revealed word by word.
 *
 * Two things are deliberately NOT gated behind the scroll: the escalation
 * banner and the disclaimer. Someone who never scrolls past the first screen
 * still has to see how soon to act.
 */

import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

import {
  ESCALATION_LABEL,
  headlineRow,
  parseRange,
  toNumber,
  type Analysis,
} from "../lib/api";

gsap.registerPlugin(useGSAP, ScrollTrigger);

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** A flat sparkline drawn into the heading, sized by the chip it sits in. */
function Sparkline() {
  return (
    <span className="inline-chip" aria-hidden="true">
      <svg viewBox="0 0 100 32" preserveAspectRatio="none">
        <polyline
          points="0,24 14,20 28,22 42,9 56,14 70,26 84,28 100,30"
          fill="none"
          stroke="rgba(255,255,255,0.75)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}

export default function Story({ data }: { data: Analysis }) {
  const root = useRef<HTMLDivElement>(null);

  const rows = data.lab_results;
  const abnormal = rows.filter((r) => r.flag === "RED" || r.flag === "AMBER");
  const headline = headlineRow(rows);
  const level = data.reasoning.escalation_level;

  const headlineRange = headline ? parseRange(headline.reference_range) : null;
  const headlineValue = headline ? toNumber(headline.observed_value) : null;

  // Where the marker sits on the track. The normal band occupies the middle
  // 60%, so a value outside it still lands on screen however far out it is.
  let markerPct = 50;
  if (headlineRange && headlineValue !== null) {
    const [low, high] = headlineRange;
    const span = high - low || 1;
    const t = (headlineValue - low) / span;
    markerPct = Math.max(3, Math.min(97, 20 + t * 60));
  }

  useGSAP(
    () => {
      const reduced = prefersReducedMotion();

      if (reduced) {
        // Land everything in its final state; no scroll choreography at all.
        gsap.set(".js-count, .js-dot, .js-spot, .js-bar, .js-marker", {
          opacity: 1,
          clearProps: "transform",
        });
        gsap.set(".js-word", { opacity: 1 });
        gsap.set(".js-normal-band", { scaleX: 1 });
        return;
      }

      // Panel 1 — the count rolls up once, then the panel eases away.
      const counter = { n: 0 };
      gsap.to(counter, {
        n: abnormal.length,
        duration: 1.1,
        ease: "power2.out",
        delay: 0.25,
        onUpdate() {
          const el = root.current?.querySelector(".js-count");
          if (el) el.textContent = String(Math.round(counter.n));
        },
      });

      gsap.timeline({
        scrollTrigger: {
          trigger: ".js-panel-1",
          start: "top top",
          end: "+=90%",
          pin: true,
          scrub: 0.6,
        },
      })
        .to(".js-panel-1-inner", { scale: 0.94, opacity: 0, ease: "none" });

      // Panel 2 — the dots scatter, then sort themselves into their flags.
      gsap.timeline({
        scrollTrigger: {
          trigger: ".js-panel-2",
          start: "top top",
          end: "+=140%",
          pin: true,
          scrub: 0.7,
        },
      })
        .from(".js-dot", {
          x: () => gsap.utils.random(-260, 260),
          y: () => gsap.utils.random(-200, 200),
          scale: 0.2,
          opacity: 0.1,
          stagger: { amount: 0.7, from: "random" },
          ease: "power2.out",
        })
        .to(".js-dot", { opacity: 1, scale: 1, stagger: { amount: 0.35 } }, "<0.35");

      // Panel 3 — the single worst value, with its interval drawn underneath.
      gsap.timeline({
        scrollTrigger: {
          trigger: ".js-panel-3",
          start: "top top",
          end: "+=150%",
          pin: true,
          scrub: 0.7,
        },
      })
        .from(".js-spot", { scale: 0.62, opacity: 0, ease: "power2.out" })
        .from(".js-normal-band", { scaleX: 0, transformOrigin: "left center" }, "<0.3")
        .from(".js-marker", { left: "20%", opacity: 0, ease: "power2.inOut" }, "<0.25")
        .from(".js-spot-meta", { y: 24, opacity: 0 }, "<0.2");

      // Panel 4 — the summary scrubs in a word at a time.
      gsap.to(".js-word", {
        opacity: 1,
        stagger: 0.55,
        ease: "none",
        scrollTrigger: {
          trigger: ".js-panel-4",
          start: "top 75%",
          end: "bottom 65%",
          scrub: 0.5,
        },
      });

      // The finding cards stack up from below as they arrive.
      ScrollTrigger.batch(".js-finding", {
        start: "top 88%",
        onEnter: (batch) =>
          gsap.to(batch, {
            y: 0,
            opacity: 1,
            duration: 0.75,
            stagger: 0.09,
            ease: "power3.out",
            overwrite: true,
          }),
      });
      gsap.set(".js-finding", { y: 48, opacity: 0 });
    },
    { scope: root, dependencies: [data.report_id] }
  );

  const summaryWords = data.summary.split(/\s+/).filter(Boolean);

  return (
    <div ref={root}>
      {/* Panel 1 — how much needs attention, and how soon */}
      <section className="panel js-panel-1">
        <div className="js-panel-1-inner" style={{ textAlign: "center" }}>
          <div className={`escalation esc-${level}`}>
            {ESCALATION_LABEL[level]}
          </div>

          <h1 className="headline" style={{ marginTop: "2.2rem" }}>
            <em className="js-count">0</em> of {rows.length} results
            <Sparkline />
            need a look
          </h1>

          {data.reasoning.escalation_reasons.length > 0 && (
            <p className="lede" style={{ marginTop: "1.8rem" }}>
              {data.reasoning.escalation_reasons[0].detail}
            </p>
          )}
        </div>
      </section>

      {/* Panel 2 — every result at once, sorting itself */}
      <section className="panel js-panel-2">
        <div style={{ textAlign: "center", width: "100%" }}>
          <p className="eyebrow">Every result on the report</p>
          <h2 className="headline" style={{ fontSize: "clamp(1.9rem, 4vw, 3.4rem)" }}>
            {data.grounding.flag_summary.GREEN} of {rows.length} are where
            they should be
          </h2>

          <div className="dotfield">
            {rows.map((row, i) => (
              <span
                key={`${row.test_name}-${i}`}
                className="dot js-dot"
                data-flag={row.flag}
                title={`${row.standard_name ?? row.test_name}: ${row.observed_value ?? "-"}`}
              />
            ))}
          </div>

          <div className="legend">
            <span><i className="swatch" style={{ background: "var(--green)" }} /> In range</span>
            <span><i className="swatch" style={{ background: "var(--amber)" }} /> Borderline</span>
            <span><i className="swatch" style={{ background: "var(--red)" }} /> Outside range</span>
            <span><i className="swatch" style={{ background: "var(--grey)" }} /> Not evaluated</span>
          </div>
        </div>
      </section>

      {/* Panel 3 — the one value most worth seeing alone */}
      {headline && (
        <section className="panel js-panel-3">
          <div style={{ textAlign: "center", width: "100%" }}>
            <p className="eyebrow">Furthest outside its usual range</p>

            <div className="spotlight-value js-spot">
              {String(headline.observed_value ?? "-")}
            </div>

            <div className="js-spot-meta">
              <p className="spotlight-name">
                {headline.standard_name ?? headline.test_name}
                {headline.unit ? ` (${headline.unit})` : ""}
              </p>
              {headline.explanation && (
                <p className="spotlight-sub">{headline.explanation}</p>
              )}

              {headlineRange && (
                <div className="range">
                  <div className="range-track">
                    <span
                      className="range-normal js-normal-band"
                      style={{ left: "20%", width: "60%" }}
                    />
                    <span
                      className="range-marker js-marker"
                      style={{ left: `${markerPct}%` }}
                    />
                  </div>
                  <div className="range-ends">
                    <span>{headlineRange[0]}</span>
                    <span>usual range</span>
                    <span>{headlineRange[1]}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Panel 4 — the summary, read one word at a time */}
      <section className="panel js-panel-4">
        <div className="reveal">
          {summaryWords.map((word, i) => (
            <span className="w js-word" key={i}>
              {word}
            </span>
          ))}
        </div>
      </section>

      {/* The findings themselves */}
      {abnormal.length > 0 && (
        <section className="section wrap">
          <div className="section-head">
            <h2>What came back outside the range</h2>
            <p>
              Each line below is a result the lab printed, checked against the
              interval on your own report.
            </p>
          </div>

          <div className="stack">
            {abnormal.map((row, i) => (
              <article
                className="finding js-finding"
                data-flag={row.flag}
                key={`${row.test_name}-${i}`}
              >
                <div className="finding-top">
                  <h3 className="finding-name">
                    {row.standard_name ?? row.test_name}
                  </h3>
                  <span className="finding-value">
                    {String(row.observed_value ?? "-")}
                    {row.unit ? ` ${row.unit}` : ""}
                  </span>
                </div>
                <p className="finding-body">
                  {row.explanation ? `${row.explanation} ` : ""}
                  The usual range is {row.reference_range ?? "not printed"}.
                </p>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
