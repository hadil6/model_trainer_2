import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
const ML_PARTICLES = [
    { term: "∇ loss = 0.023", x: 4, delay: 0, dur: 22 },
    { term: "LoRA rank=16", x: 11, delay: 5, dur: 27 },
    { term: "lr = 2e-4", x: 19, delay: 9, dur: 20 },
    { term: "epoch 3 / 6", x: 28, delay: 2, dur: 25 },
    { term: "ROUGE-1: 0.48", x: 38, delay: 13, dur: 23 },
    { term: "QLoRA 4-bit", x: 49, delay: 6, dur: 28 },
    { term: "heads × 8", x: 57, delay: 16, dur: 21 },
    { term: "batch_size = 4", x: 66, delay: 1, dur: 24 },
    { term: "hidden = 512", x: 74, delay: 10, dur: 19 },
    { term: "BLEU-4: 0.12", x: 83, delay: 18, dur: 26 },
    { term: "∂L / ∂w", x: 90, delay: 4, dur: 22 },
    { term: "softmax(z)", x: 96, delay: 14, dur: 20 },
    { term: "fine-tuning ✓", x: 7, delay: 20, dur: 28 },
    { term: "dropout = 0.05", x: 23, delay: 7, dur: 23 },
    { term: "eval_loss ↓", x: 44, delay: 17, dur: 25 },
    { term: "adapter saved", x: 61, delay: 3, dur: 21 },
    { term: "θ* = argmin L", x: 78, delay: 11, dur: 24 },
    { term: "PEFT · DoRA", x: 33, delay: 22, dur: 26 },
];
export function FloatingMLBackground() {
    return (_jsx("div", { className: "ml-bg", "aria-hidden": "true", children: ML_PARTICLES.map((p, i) => (_jsx("span", { className: "ml-particle", style: {
                left: `${p.x}%`,
                animationDelay: `${p.delay}s`,
                animationDuration: `${p.dur}s`,
            }, children: p.term }, i))) }));
}
function NetSVG({ cls, animOffset = 0 }) {
    const L1 = [{ x: 50, y: 35 }, { x: 50, y: 100 }, { x: 50, y: 165 }];
    const L2 = [{ x: 150, y: 18 }, { x: 150, y: 68 }, { x: 150, y: 128 }, { x: 150, y: 178 }];
    const L3 = [{ x: 250, y: 58 }, { x: 250, y: 138 }];
    const conns = [];
    let ci = 0;
    L1.forEach(a => L2.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
    L2.forEach(a => L3.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
    const nodes = [...L1, ...L2, ...L3];
    return (_jsxs("svg", { className: cls, viewBox: "0 0 300 200", "aria-hidden": "true", children: [conns.map(c => (_jsx("line", { x1: c.x1, y1: c.y1, x2: c.x2, y2: c.y2, stroke: "rgba(124,58,237,0.13)", strokeWidth: "1.5", strokeDasharray: "5 4", className: "net-line-anim", style: { animationDelay: `${((c.i * 0.2) + animOffset) % 2.2}s` } }, c.i))), nodes.map((n, i) => (_jsx("circle", { cx: n.x, cy: n.y, r: "10", fill: "rgba(124,58,237,0.07)", stroke: "rgba(124,58,237,0.28)", strokeWidth: "1.5", className: "net-node", style: { animationDelay: `${((i * 0.35) + animOffset) % 2.8}s` } }, i)))] }));
}
export function NeuralNetPageBackground() {
    return (_jsxs(_Fragment, { children: [_jsx(NetSVG, { cls: "nn-bg nn-bg-right-mid", animOffset: 0 }), _jsx(NetSVG, { cls: "nn-bg nn-bg-left-mid", animOffset: 0.7 }), _jsx(NetSVG, { cls: "nn-bg nn-bg-right-top", animOffset: 1.1 }), _jsx(NetSVG, { cls: "nn-bg nn-bg-left-bot", animOffset: 1.5 }), _jsx(NetSVG, { cls: "nn-bg nn-bg-right-bot", animOffset: 0.4 }), _jsx(NetSVG, { cls: "nn-bg nn-bg-center-top", animOffset: 1.8 })] }));
}
export function NeuralNetDecoration() {
    const L1 = [{ x: 50, y: 35 }, { x: 50, y: 100 }, { x: 50, y: 165 }];
    const L2 = [{ x: 150, y: 18 }, { x: 150, y: 68 }, { x: 150, y: 128 }, { x: 150, y: 178 }];
    const L3 = [{ x: 250, y: 58 }, { x: 250, y: 138 }];
    const conns = [];
    let ci = 0;
    L1.forEach(a => L2.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
    L2.forEach(a => L3.forEach(b => conns.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, i: ci++ })));
    const nodes = [...L1, ...L2, ...L3];
    return (_jsxs("svg", { className: "neural-net-svg", viewBox: "0 0 300 200", "aria-hidden": "true", children: [conns.map(c => (_jsx("line", { x1: c.x1, y1: c.y1, x2: c.x2, y2: c.y2, stroke: "rgba(255,255,255,0.22)", strokeWidth: "1", strokeDasharray: "5 4", className: "net-line-anim", style: { animationDelay: `${(c.i * 0.2) % 2.2}s` } }, c.i))), nodes.map((n, i) => (_jsx("circle", { cx: n.x, cy: n.y, r: "9", fill: "rgba(255,255,255,0.12)", stroke: "rgba(255,255,255,0.45)", strokeWidth: "1.5", className: "net-node", style: { animationDelay: `${(i * 0.35) % 2.8}s` } }, i)))] }));
}
