const ML_PARTICLES = [
  { term: "∇ loss = 0.023",  x: 4,  delay: 0,  dur: 22 },
  { term: "LoRA rank=16",    x: 11, delay: 5,  dur: 27 },
  { term: "lr = 2e-4",       x: 19, delay: 9,  dur: 20 },
  { term: "epoch 3 / 6",     x: 28, delay: 2,  dur: 25 },
  { term: "ROUGE-1: 0.48",   x: 38, delay: 13, dur: 23 },
  { term: "QLoRA 4-bit",     x: 49, delay: 6,  dur: 28 },
  { term: "heads × 8",       x: 57, delay: 16, dur: 21 },
  { term: "batch_size = 4",  x: 66, delay: 1,  dur: 24 },
  { term: "hidden = 512",    x: 74, delay: 10, dur: 19 },
  { term: "BLEU-4: 0.12",    x: 83, delay: 18, dur: 26 },
  { term: "∂L / ∂w",         x: 90, delay: 4,  dur: 22 },
  { term: "softmax(z)",      x: 96, delay: 14, dur: 20 },
  { term: "fine-tuning ✓",   x: 7,  delay: 20, dur: 28 },
  { term: "dropout = 0.05",  x: 23, delay: 7,  dur: 23 },
  { term: "eval_loss ↓",     x: 44, delay: 17, dur: 25 },
  { term: "adapter saved",   x: 61, delay: 3,  dur: 21 },
  { term: "θ* = argmin L",   x: 78, delay: 11, dur: 24 },
  { term: "PEFT · DoRA",     x: 33, delay: 22, dur: 26 },
];

export function FloatingMLBackground() {
  return (
    <div className="ml-bg" aria-hidden="true">
      {ML_PARTICLES.map((p, i) => (
        <span
          key={i}
          className="ml-particle"
          style={{
            left: `${p.x}%`,
            animationDelay: `${p.delay}s`,
            animationDuration: `${p.dur}s`,
          }}
        >
          {p.term}
        </span>
      ))}
    </div>
  );
}

function NetSVG({ cls, animOffset = 0 }: { cls: string; animOffset?: number }) {
  const L1 = [{ x: 50, y: 35 }, { x: 50, y: 100 }, { x: 50, y: 165 }];
  const L2 = [{ x: 150, y: 18 }, { x: 150, y: 68 }, { x: 150, y: 128 }, { x: 150, y: 178 }];
  const L3 = [{ x: 250, y: 58 }, { x: 250, y: 138 }];

  type Conn = { x1: number; y1: number; x2: number; y2: number; i: number };
  const conns: Conn[] = [];
  let ci = 0;
  L1.forEach(a => L2.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
  L2.forEach(a => L3.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
  const nodes = [...L1, ...L2, ...L3];

  return (
    <svg className={cls} viewBox="0 0 300 200" aria-hidden="true">
      {conns.map(c => (
        <line
          key={c.i}
          x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2}
          stroke="rgba(124,58,237,0.13)" strokeWidth="1.5"
          strokeDasharray="5 4"
          className="net-line-anim"
          style={{ animationDelay: `${((c.i * 0.2) + animOffset) % 2.2}s` }}
        />
      ))}
      {nodes.map((n, i) => (
        <circle
          key={i} cx={n.x} cy={n.y} r="10"
          fill="rgba(124,58,237,0.07)"
          stroke="rgba(124,58,237,0.28)"
          strokeWidth="1.5"
          className="net-node"
          style={{ animationDelay: `${((i * 0.35) + animOffset) % 2.8}s` }}
        />
      ))}
    </svg>
  );
}

export function NeuralNetPageBackground() {
  return (
    <>
      <NetSVG cls="nn-bg nn-bg-right-mid"  animOffset={0}   />
      <NetSVG cls="nn-bg nn-bg-left-mid"   animOffset={0.7} />
      <NetSVG cls="nn-bg nn-bg-right-top"  animOffset={1.1} />
      <NetSVG cls="nn-bg nn-bg-left-bot"   animOffset={1.5} />
      <NetSVG cls="nn-bg nn-bg-right-bot"  animOffset={0.4} />
      <NetSVG cls="nn-bg nn-bg-center-top" animOffset={1.8} />
    </>
  );
}

export function NeuralNetDecoration() {
  const L1 = [{ x: 50, y: 35 }, { x: 50, y: 100 }, { x: 50, y: 165 }];
  const L2 = [{ x: 150, y: 18 }, { x: 150, y: 68 }, { x: 150, y: 128 }, { x: 150, y: 178 }];
  const L3 = [{ x: 250, y: 58 }, { x: 250, y: 138 }];

  type Conn = { x1: number; y1: number; x2: number; y2: number; i: number };
  const conns: Conn[] = [];
  let ci = 0;
  L1.forEach(a => L2.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
  L2.forEach(a => L3.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
  const nodes = [...L1, ...L2, ...L3];

  return (
    <svg className="neural-net-svg" viewBox="0 0 300 200" aria-hidden="true">
      {conns.map(c => (
        <line
          key={c.i}
          x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2}
          stroke="rgba(255,255,255,0.22)" strokeWidth="1"
          strokeDasharray="5 4"
          className="net-line-anim"
          style={{ animationDelay: `${(c.i * 0.2) % 2.2}s` }}
        />
      ))}
      {nodes.map((n, i) => (
        <circle
          key={i} cx={n.x} cy={n.y} r="9"
          fill="rgba(255,255,255,0.12)"
          stroke="rgba(255,255,255,0.45)"
          strokeWidth="1.5"
          className="net-node"
          style={{ animationDelay: `${(i * 0.35) % 2.8}s` }}
        />
      ))}
    </svg>
  );
}
