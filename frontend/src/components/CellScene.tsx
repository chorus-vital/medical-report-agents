/**
 * A count-accurate 3D view of one cell population.
 *
 * Two identical sample volumes sit side by side: what the report measured, and
 * what the low end of the usual range would look like. Density is proportional,
 * so the gap you see is the gap in the number — this is the data rendered, not
 * an illustration of it.
 *
 * Deliberately anatomical, never pathological. It shows what the cell is and
 * how many there are. It never shows what a low count does to you, because
 * that is a claim about cause, and nothing in this pipeline can verify a
 * picture the way the verifier checks the prose.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

import type { LabRow } from "../lib/api";
import { parseRange, toNumber } from "../lib/api";

type CellKind = "leukocyte" | "erythrocyte" | "platelet";

/** Which analytes we can honestly draw. Anything else gets no panel at all. */
const CELL_BY_LOINC: Record<string, CellKind> = {
  "6690-2": "leukocyte", // Total WBC Count
  "751-8": "leukocyte", // Absolute Neutrophil Count
  "731-0": "leukocyte", // Absolute Lymphocyte Count
  "742-7": "leukocyte", // Absolute Monocyte Count
  "711-2": "leukocyte", // Absolute Eosinophil Count
  "704-7": "leukocyte", // Absolute Basophil Count
  "789-8": "erythrocyte", // Red Blood Cell Count
  "777-3": "platelet", // Platelet Count
};

const CELL_COPY: Record<CellKind, { name: string; body: string }> = {
  leukocyte: {
    name: "White blood cell",
    body: "Larger than a red cell and far less numerous. Several types exist; they circulate and move out of the bloodstream into tissue.",
  },
  erythrocyte: {
    name: "Red blood cell",
    body: "A flexible biconcave disc with no nucleus, shaped to squeeze through the narrowest vessels. It carries haemoglobin.",
  },
  platelet: {
    name: "Platelet",
    body: "The smallest of the three, a cell fragment rather than a whole cell. Platelets clump together where a vessel wall is damaged.",
  },
};

// How many instances represent the low end of the usual range. Both volumes use
// the same scale, so the ratio on screen equals the ratio in the numbers.
const REFERENCE_INSTANCES = 200;
const MAX_INSTANCES = 900;

function geometryFor(kind: CellKind): THREE.BufferGeometry {
  if (kind === "erythrocyte") {
    // Biconcave-ish: a flattened sphere reads correctly at this scale.
    const g = new THREE.SphereGeometry(0.5, 20, 14);
    g.scale(1, 0.34, 1);
    return g;
  }
  if (kind === "platelet") {
    return new THREE.IcosahedronGeometry(0.26, 0);
  }
  // Leukocyte: lumpy and round.
  return new THREE.IcosahedronGeometry(0.55, 2);
}

function colourFor(kind: CellKind): string {
  if (kind === "erythrocyte") return "#c0392f";
  if (kind === "platelet") return "#d8c07a";
  return "#dfe6f2";
}

interface CloudProps {
  count: number;
  kind: CellKind;
  ghost: boolean;
  animate: boolean;
  offsetX: number;
}

function Cloud({ count, kind, ghost, animate, offsetX }: CloudProps) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const geometry = useMemo(() => geometryFor(kind), [kind]);

  // Fixed seeds so the cloud does not reshuffle on every React render.
  const seeds = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => ({
        x: (Math.sin(i * 12.9898) * 43758.5453) % 1,
        y: (Math.sin(i * 78.233) * 12345.6789) % 1,
        z: (Math.sin(i * 39.425) * 24634.6345) % 1,
        phase: (i % 40) / 40,
        spin: ((i % 7) - 3) * 0.12,
      })),
    [count]
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);

  const place = (time: number) => {
    if (!mesh.current) return;
    for (let i = 0; i < count; i++) {
      const s = seeds[i];
      const bob = animate ? Math.sin(time * 0.5 + s.phase * Math.PI * 2) * 0.12 : 0;
      dummy.position.set(
        offsetX + (s.x - 0.5) * 5.2,
        (s.y - 0.5) * 5.2 + bob,
        (s.z - 0.5) * 5.2
      );
      dummy.rotation.set(
        s.x * Math.PI * 2 + (animate ? time * s.spin * 0.25 : 0),
        s.y * Math.PI * 2,
        s.z * Math.PI * 2
      );
      const scale = 0.75 + s.z * 0.5;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.current.setMatrixAt(i, dummy.matrix);
    }
    mesh.current.instanceMatrix.needsUpdate = true;
  };

  useEffect(() => {
    place(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, offsetX, animate]);

  useFrame((state) => {
    if (animate) place(state.clock.elapsedTime);
  });

  return (
    <instancedMesh
      ref={mesh}
      args={[geometry, undefined, Math.max(count, 1)]}
      frustumCulled={false}
    >
      <meshStandardMaterial
        color={colourFor(kind)}
        roughness={0.42}
        metalness={0.04}
        transparent
        opacity={ghost ? 0.14 : 0.95}
      />
    </instancedMesh>
  );
}

function Scene({
  kind,
  actual,
  expected,
  animate,
}: {
  kind: CellKind;
  actual: number;
  expected: number;
  animate: boolean;
}) {
  const group = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (animate && group.current) {
      group.current.rotation.y = state.clock.elapsedTime * 0.06;
    }
  });

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[6, 8, 5]} intensity={1.5} />
      <directionalLight position={[-7, -3, -5]} intensity={0.5} color="#6f8dff" />

      <group ref={group}>
        <Cloud count={expected} kind={kind} ghost animate={animate} offsetX={3.6} />
        <Cloud count={actual} kind={kind} ghost={false} animate={animate} offsetX={-3.6} />
      </group>
    </>
  );
}

export default function CellScene({ rows }: { rows: LabRow[] }) {
  const [visible, setVisible] = useState(false);
  const host = useRef<HTMLDivElement>(null);

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Only the abnormal rows we can actually draw, furthest outside its interval
  // first — so the one that opens is the one most worth looking at.
  const candidates = useMemo(() => {
    const deviation = (r: LabRow) => {
      const range = parseRange(r.reference_range);
      const value = toNumber(r.observed_value);
      if (!range || value === null) return 0;
      const [low, high] = range;
      const span = high - low || 1;
      return value < low ? (low - value) / span : (value - high) / span;
    };

    return rows
      .filter(
        (r) =>
          (r.flag === "RED" || r.flag === "AMBER") &&
          r.loinc_code &&
          CELL_BY_LOINC[r.loinc_code] &&
          toNumber(r.observed_value) !== null &&
          parseRange(r.reference_range) !== null
      )
      .sort((a, b) => deviation(b) - deviation(a));
  }, [rows]);

  const [pick, setPick] = useState(0);

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const io = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && setVisible(true),
      { rootMargin: "200px" }
    );
    io.observe(node);
    return () => io.disconnect();
  }, []);

  if (!candidates.length) return null;

  const row = candidates[Math.min(pick, candidates.length - 1)];
  const kind = CELL_BY_LOINC[row.loinc_code!];
  const value = toNumber(row.observed_value)!;
  const [low] = parseRange(row.reference_range)!;

  const ratio = low > 0 ? value / low : 1;
  const actual = Math.max(1, Math.min(MAX_INSTANCES, Math.round(REFERENCE_INSTANCES * ratio)));
  const expected = REFERENCE_INSTANCES;
  const perInstance = low / REFERENCE_INSTANCES;

  return (
    <section className="section wrap" ref={host}>
      <div className="section-head">
        <h2>What that looks like</h2>
        <p>
          The same volume of blood, twice. On the left, the number your report
          measured. On the right, the low end of the usual range. Density is
          proportional, so the difference you see is the difference in the count.
        </p>
      </div>

      {candidates.length > 1 && (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.4rem" }}>
          {candidates.map((c, i) => (
            <button
              key={c.test_name + i}
              type="button"
              className={i === pick ? "btn btn-solid" : "btn btn-ghost"}
              style={{ padding: "0.5rem 1rem", fontSize: "0.78rem" }}
              onClick={() => setPick(i)}
            >
              {c.standard_name ?? c.test_name}
            </button>
          ))}
        </div>
      )}

      <div className="scene">
        <div className="scene-canvas">
          {visible && (
            <Canvas
              camera={{ position: [0, 0.5, 13], fov: 42 }}
              dpr={[1, 1.75]}
              frameloop={reduced ? "demand" : "always"}
            >
              <Scene kind={kind} actual={actual} expected={expected} animate={!reduced} />
            </Canvas>
          )}
        </div>

        <div className="scene-labels">
          <span className="scene-label scene-label-actual">
            Yours — {row.observed_value} {row.unit ?? ""}
          </span>
          <span className="scene-label scene-label-ghost">
            Usual minimum — {low} {row.unit ?? ""}
          </span>
        </div>
      </div>

      <div className="scene-note">
        <p>
          <strong>{CELL_COPY[kind].name}.</strong> {CELL_COPY[kind].body}
        </p>
        <p className="scene-scale">
          One shape stands for about {Math.round(perInstance).toLocaleString()}{" "}
          {row.unit ? `${row.unit}` : "units"}. Shapes are representative, not
          to scale with each other.
        </p>
      </div>
    </section>
  );
}
