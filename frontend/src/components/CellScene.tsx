/**
 * A count-accurate, explorable 3D view of one cell population.
 *
 * Two identical sample volumes: what the report measured, and what the low end
 * of the usual range would look like. Density is proportional, so the gap you
 * see is the gap in the number — this is the data rendered, not an illustration
 * of it. Side by side, or overlaid so the shortfall reads as empty space.
 *
 * Deliberately anatomical, never pathological. It shows what the cell is and
 * how many there are. It never shows what a low count does to you, because that
 * is a claim about cause, and nothing in this pipeline can verify a picture the
 * way the verifier checks the prose.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
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

const CELL_COPY: Record<CellKind, { name: string; body: string; size: string }> = {
  leukocyte: {
    name: "White blood cell",
    body: "Larger than a red cell and far less numerous. Several types exist; they circulate and move out of the bloodstream into tissue.",
    size: "about 12-15 micrometres across",
  },
  erythrocyte: {
    name: "Red blood cell",
    body: "A flexible biconcave disc with no nucleus, shaped to squeeze through the narrowest vessels. It carries haemoglobin.",
    size: "about 7.5 micrometres across, 2 thick",
  },
  platelet: {
    name: "Platelet",
    body: "The smallest of the three, a cell fragment rather than a whole cell. Platelets clump together where a vessel wall is damaged.",
    size: "about 2-3 micrometres across",
  },
};

// How many instances represent the low end of the usual range. Both volumes use
// the same scale, so the ratio on screen equals the ratio in the numbers.
const REFERENCE_INSTANCES = 200;
const MAX_INSTANCES = 900;
const BOX = 5.4;
const SPLIT_X = 3.5;

/* ── geometry ─────────────────────────────────────────────────────────────── */

/** Push vertices around a little so a membrane does not read as a billiard ball. */
function roughen(geometry: THREE.BufferGeometry, amount: number, freq: number) {
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n =
      Math.sin(v.x * freq) * Math.sin(v.y * (freq * 0.9)) * Math.sin(v.z * (freq * 1.1));
    v.multiplyScalar(1 + n * amount);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * A real biconcave disc, revolved from its profile — thin at the centre, thick
 * at the rim. A flattened sphere reads as a lentil and gets the shape wrong.
 */
function biconcave(radius: number) {
  const halfThickness = (r: number) =>
    0.5 * Math.sqrt(Math.max(0, 1 - r * r)) * (0.21 + 2.0 * r * r - 1.12 * r ** 4);

  const steps = 24;
  const points: THREE.Vector2[] = [];
  for (let i = 0; i <= steps; i++) {
    const r = i / steps;
    points.push(new THREE.Vector2(r * radius, -halfThickness(r) * radius));
  }
  for (let i = steps; i >= 0; i--) {
    const r = i / steps;
    points.push(new THREE.Vector2(r * radius, halfThickness(r) * radius));
  }
  const g = new THREE.LatheGeometry(points, 28);
  g.computeVertexNormals();
  return g;
}

function geometryFor(kind: CellKind): THREE.BufferGeometry {
  if (kind === "erythrocyte") return biconcave(0.55);
  if (kind === "platelet") {
    const g = roughen(new THREE.IcosahedronGeometry(0.3, 2), 0.22, 9);
    g.scale(1, 0.62, 1);
    return g;
  }
  return roughen(new THREE.IcosahedronGeometry(0.55, 3), 0.15, 6.5);
}

/** Rough bounding radius, for the cheap hover test. */
const HIT_RADIUS: Record<CellKind, number> = {
  leukocyte: 0.62,
  erythrocyte: 0.58,
  platelet: 0.34,
};

const PALETTE: Record<CellKind, { base: string; sheen: string }> = {
  leukocyte: { base: "#e9eff9", sheen: "#8fa8ff" },
  erythrocyte: { base: "#bb332b", sheen: "#ff8d7d" },
  platelet: { base: "#dcc27f", sheen: "#ffe9a8" },
};

/* ── clouds ───────────────────────────────────────────────────────────────── */

interface CloudProps {
  count: number;
  kind: CellKind;
  ghost: boolean;
  animate: boolean;
  offsetX: number;
  hovered?: number | null;
  onHover?: (id: number | null) => void;
}

function Cloud({ count, kind, ghost, animate, offsetX, hovered, onHover }: CloudProps) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const geometry = useMemo(() => geometryFor(kind), [kind]);
  const palette = PALETTE[kind];

  // Fixed pseudo-random seeds, so the cloud does not reshuffle every render.
  const seeds = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => {
        const f = (a: number, b: number) => ((Math.sin(i * a) * b) % 1 + 1) % 1;
        return {
          x: f(12.9898, 43758.5453),
          y: f(78.233, 12345.6789),
          z: f(39.425, 24634.6345),
          phase: f(5.117, 9871.23),
          spin: (f(3.77, 4412.1) - 0.5) * 0.5,
        };
      }),
    [count]
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);

  /**
   * Raycast against one sphere per instance instead of the real mesh. The
   * leukocyte alone is 1,280 triangles; testing 900 of those on every pointer
   * move would cost a million triangle intersections per frame.
   */
  useEffect(() => {
    const node = mesh.current;
    if (!node || ghost) return;
    const m = new THREE.Matrix4();
    const sphere = new THREE.Sphere();
    const centre = new THREE.Vector3();
    const radius = HIT_RADIUS[kind];

    node.raycast = function (raycaster, intersects) {
      for (let i = 0; i < this.count; i++) {
        this.getMatrixAt(i, m);
        centre.setFromMatrixPosition(m).applyMatrix4(this.matrixWorld);
        sphere.set(centre, radius * m.getMaxScaleOnAxis() * 1.15);
        if (raycaster.ray.intersectsSphere(sphere)) {
          intersects.push({
            distance: raycaster.ray.origin.distanceTo(centre),
            point: centre.clone(),
            object: this,
            instanceId: i,
          } as THREE.Intersection);
        }
      }
    };
  }, [kind, ghost, count]);

  const place = (time: number) => {
    const node = mesh.current;
    if (!node) return;
    for (let i = 0; i < count; i++) {
      const s = seeds[i];
      const bob = animate ? Math.sin(time * 0.45 + s.phase * Math.PI * 2) * 0.13 : 0;
      const drift = animate ? Math.cos(time * 0.3 + s.x * 6.28) * 0.07 : 0;

      dummy.position.set(
        offsetX + (s.x - 0.5) * BOX + drift,
        (s.y - 0.5) * BOX + bob,
        (s.z - 0.5) * BOX
      );
      dummy.rotation.set(
        s.x * Math.PI * 2 + (animate ? time * s.spin * 0.3 : 0),
        s.y * Math.PI * 2 + (animate ? time * s.spin * 0.2 : 0),
        s.z * Math.PI * 2
      );
      const base = 0.72 + s.z * 0.5;
      dummy.scale.setScalar(hovered === i && !ghost ? base * 2.1 : base);
      dummy.updateMatrix();
      node.setMatrixAt(i, dummy.matrix);
    }
    node.instanceMatrix.needsUpdate = true;
  };

  useEffect(() => {
    place(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, offsetX, animate, hovered]);

  useFrame((state) => {
    if (animate) place(state.clock.elapsedTime);
  });

  return (
    <instancedMesh
      ref={mesh}
      args={[geometry, undefined, Math.max(count, 1)]}
      frustumCulled={false}
      onPointerMove={
        ghost
          ? undefined
          : (e) => {
              e.stopPropagation();
              if (typeof e.instanceId === "number") onHover?.(e.instanceId);
            }
      }
      onPointerOut={ghost ? undefined : () => onHover?.(null)}
    >
      {ghost ? (
        <meshBasicMaterial
          color={palette.base}
          wireframe
          transparent
          opacity={0.13}
          depthWrite={false}
        />
      ) : (
        <meshPhysicalMaterial
          color={palette.base}
          roughness={0.38}
          metalness={0.02}
          clearcoat={0.45}
          clearcoatRoughness={0.35}
          sheen={1}
          sheenColor={palette.sheen}
          sheenRoughness={0.5}
        />
      )}
    </instancedMesh>
  );
}

/** The sample volume itself, so "the same amount of blood" is visible. */
function VolumeFrame({ offsetX }: { offsetX: number }) {
  const geometry = useMemo(
    () => new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX + 1.2, BOX + 1.2, BOX + 1.2)),
    []
  );
  return (
    <lineSegments geometry={geometry} position={[offsetX, 0, 0]}>
      <lineBasicMaterial color="#39445a" transparent opacity={0.4} />
    </lineSegments>
  );
}

function Controls({ autoRotate, resetKey }: { autoRotate: boolean; resetKey: number }) {
  const { camera, gl, invalidate } = useThree();
  const controls = useRef<OrbitControls | null>(null);

  useEffect(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.enablePan = false;
    c.minDistance = 8;
    c.maxDistance = 24;
    c.rotateSpeed = 0.55;
    c.zoomSpeed = 0.7;
    c.autoRotateSpeed = 0.5;
    // Wrapped rather than passed directly: invalidate() takes an optional frame
    // count, which does not match the listener signature three expects.
    const redraw = () => invalidate();
    c.addEventListener("change", redraw);
    controls.current = c;
    return () => {
      c.removeEventListener("change", redraw);
      c.dispose();
    };
  }, [camera, gl, invalidate]);

  useEffect(() => {
    camera.position.set(0, 1, 16);
    controls.current?.target.set(0, 0, 0);
    controls.current?.update();
    invalidate();
  }, [resetKey, camera, invalidate]);

  useFrame(() => {
    const c = controls.current;
    if (!c) return;
    c.autoRotate = autoRotate;
    c.update();
  });

  return null;
}

function Scene({
  kind,
  actual,
  expected,
  animate,
  overlay,
  autoRotate,
  resetKey,
  hovered,
  onHover,
}: {
  kind: CellKind;
  actual: number;
  expected: number;
  animate: boolean;
  overlay: boolean;
  autoRotate: boolean;
  resetKey: number;
  hovered: number | null;
  onHover: (id: number | null) => void;
}) {
  const actualX = overlay ? 0 : -SPLIT_X;
  const ghostX = overlay ? 0 : SPLIT_X;

  return (
    <>
      <fogExp2 attach="fog" args={["#0a0c10", 0.038]} />

      <ambientLight intensity={0.5} />
      <directionalLight position={[7, 9, 6]} intensity={1.7} />
      <directionalLight position={[-8, -2, -6]} intensity={0.55} color="#7a92ff" />
      <pointLight position={[0, 0, 9]} intensity={28} distance={26} color="#ffffff" />

      <Controls autoRotate={autoRotate} resetKey={resetKey} />

      <VolumeFrame offsetX={actualX} />
      {!overlay && <VolumeFrame offsetX={ghostX} />}

      <Cloud count={expected} kind={kind} ghost animate={animate} offsetX={ghostX} />
      <Cloud
        count={actual}
        kind={kind}
        ghost={false}
        animate={animate}
        offsetX={actualX}
        hovered={hovered}
        onHover={onHover}
      />
    </>
  );
}

/* ── panel ────────────────────────────────────────────────────────────────── */

export default function CellScene({ rows }: { rows: LabRow[] }) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [pick, setPick] = useState(0);
  const [overlay, setOverlay] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [resetKey, setResetKey] = useState(0);
  const [hovered, setHovered] = useState<number | null>(null);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Only the abnormal rows we can draw, furthest outside its interval first —
  // so the one that opens is the one most worth looking at.
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

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const io = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && setVisible(true),
      { rootMargin: "240px" }
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
  const perInstance = low / REFERENCE_INSTANCES;
  const shortfall = Math.max(0, REFERENCE_INSTANCES - actual);

  return (
    <section className="section wrap" ref={host}>
      <div className="section-head">
        <h2>What that looks like</h2>
        <p>
          The same volume of blood, twice. Your measured count against the low
          end of the usual range, at the same scale — so the difference you see
          is the difference in the number. Drag to turn it, scroll to move closer.
        </p>
      </div>

      {candidates.length > 1 && (
        <div className="scene-picker">
          {candidates.map((c, i) => (
            <button
              key={c.test_name + i}
              type="button"
              className={i === pick ? "chip chip-on" : "chip"}
              onClick={() => {
                setPick(i);
                setHovered(null);
              }}
            >
              {c.standard_name ?? c.test_name}
            </button>
          ))}
        </div>
      )}

      <div
        className="scene"
        onPointerMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          setPointer({ x: e.clientX - box.left, y: e.clientY - box.top });
        }}
      >
        <div className="scene-canvas">
          {visible && (
            <Canvas
              camera={{ position: [0, 1, 16], fov: 42 }}
              dpr={[1, 1.75]}
              frameloop={reduced ? "demand" : "always"}
              gl={{ antialias: true }}
            >
              <Scene
                kind={kind}
                actual={actual}
                expected={REFERENCE_INSTANCES}
                animate={!reduced}
                overlay={overlay}
                autoRotate={autoRotate && !reduced}
                resetKey={resetKey}
                hovered={hovered}
                onHover={setHovered}
              />
            </Canvas>
          )}
        </div>

        {hovered !== null && (
          <div
            className="scene-tip"
            style={{ left: pointer.x, top: pointer.y }}
            aria-hidden="true"
          >
            <strong>{CELL_COPY[kind].name}</strong>
            <span>{CELL_COPY[kind].size}</span>
            <span>
              one of {actual} shown &middot; each stands for ~
              {Math.round(perInstance).toLocaleString()}
            </span>
          </div>
        )}

        <div className="scene-tools">
          <button
            type="button"
            className={overlay ? "chip chip-on" : "chip"}
            onClick={() => setOverlay((v) => !v)}
          >
            {overlay ? "Side by side" : "Overlay"}
          </button>
          <button
            type="button"
            className={autoRotate ? "chip chip-on" : "chip"}
            onClick={() => setAutoRotate((v) => !v)}
            disabled={reduced}
          >
            {autoRotate ? "Pause" : "Rotate"}
          </button>
          <button type="button" className="chip" onClick={() => setResetKey((k) => k + 1)}>
            Recentre
          </button>
        </div>

        <div className="scene-labels">
          <span className="scene-label scene-label-actual">
            Yours &mdash; {row.observed_value} {row.unit ?? ""}
          </span>
          <span className="scene-label scene-label-ghost">
            Usual minimum &mdash; {low} {row.unit ?? ""}
          </span>
        </div>
      </div>

      <div className="scene-note">
        <p>
          <strong>{CELL_COPY[kind].name}.</strong> {CELL_COPY[kind].body}
        </p>
        {shortfall > 0 && (
          <p>
            {overlay
              ? "The wireframe shells are the cells that would be there at the low end of the usual range."
              : `The right-hand volume holds ${shortfall} more shapes than the left.`}
          </p>
        )}
        <p className="scene-scale">
          One shape stands for about {Math.round(perInstance).toLocaleString()}{" "}
          {row.unit ?? "units"}. Shapes are representative, and not to scale with
          each other.
        </p>
      </div>
    </section>
  );
}
