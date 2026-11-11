// The app shell: home screen, four tabs (hash routes), pixel icons, and the hero waveform.
// Data and Stats are two views of the explorer (explorer.js). Model and Try wait for a model.
import { initExplorer, showExplorer } from "./explorer.js";

const $ = (id) => document.getElementById(id);
const SCREENS = ["home", "data", "stats", "model", "try"];
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------- pixel icons: one string per row; "#" = currentColor, other letters = palette ----------
const ICONS = {
  home: ["....#....", "...###...", "..#####..", ".#######.", "#########", ".##...##.", ".##...##.", ".##...##.", ".#######."],
  wave: bars([2, 5, 8, 4, 11, 7, 3, 9, 6, 3, 1]),
  wave2: bars([2, 5, 8, 4, 11, 7, 3, 9, 6, 3, 1], "hhhhhhaaaaa"),
  chart: ["..h......", "..h...a..", ".hh...a..", ".hhh..aa.", "hhhh.aaa.", "hhhh.aaaa", "#########"],
  chip: ["..#.#.#..", ".#######.", "##.....##", ".#.###.#.", "##.###.##", ".#.###.#.", "##.....##", ".#######.", "..#.#.#.."],
  mic: ["...###...", "..#####..", "..#####..", "..#####..", "..#####..", "#.#####.#", "#..###..#", ".#.....#.", "..#####..", "....#....", "..#####.."],
  ext: ["..#####", ".....##", "....###", "...##.#", "..##..#", ".##....", "##....."],
  dice: ["#########", "#.......#", "#.#...#.#", "#.......#", "#...#...#", "#.......#", "#.#...#.#", "#.......#", "#########"],
  barrier: [
    "..ww........ww..", "yykkyykkyykkyykk", "ykkyykkyykkyykky", "kkyykkyykkyykkyy",
    "..ww........ww..", "..ww........ww..", "yykkyykkyykkyykk", "ykkyykkyykkyykky",
    "kkyykkyykkyykkyy", "..ww........ww..", ".wwww......wwww.",
  ],
};
const PALETTE = { y: "var(--gold)", k: "#14171c", w: "#c9d3db", h: "var(--human)", a: "var(--ai)" };

function bars(heights, colors = "") {
  const H = Math.max(...heights);
  return Array.from({ length: H }, (_, r) =>
    heights.map((h, x) => (Math.abs(r - (H - 1) / 2) < h / 2 ? colors[x] || "#" : ".")).join(""));
}
function pix(rows) {
  let rects = "";
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; ) {
      const c = row[x];
      if (c === ".") { x++; continue; }
      let w = 1;
      while (row[x + w] === c) w++;
      rects += `<rect x="${x}" y="${y}" width="${w}" height="1" style="fill:${c === "#" ? "currentColor" : PALETTE[c]}"/>`;
      x += w;
    }
  });
  return `<svg viewBox="0 0 ${rows[0].length} ${rows.length}" shape-rendering="crispEdges" aria-hidden="true">${rects}</svg>`;
}
document.querySelectorAll("[data-pix]").forEach((el) => { el.innerHTML = pix(ICONS[el.dataset.pix]); });

// ---------- hero: a pixel waveform, human (blue) on the left, AI (red) on the right ----------
const hero = (() => {
  const cv = $("hero"), g = cv.getContext("2d"), W = 96, H = 30;
  cv.width = W; cv.height = H;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  let raf = 0;
  function frame(ms) {
    const t = ms / 1000, human = css("--human"), ai = css("--ai"), shadow = css("--shadow");
    g.clearRect(0, 0, W, H);
    for (let x = 1; x < W - 1; x += 2) {
      const u = x / W;
      const word = 0.25 + 0.75 * Math.abs(Math.sin(u * 9 + t * 0.9)); // syllable-like bursts
      const buzz = 0.55 + 0.45 * Math.sin(x * 1.3 + t * 7);
      const h = Math.max(2, Math.round(word * buzz * (H - 6)));
      const y = Math.round((H - h) / 2) - 1;
      g.fillStyle = shadow; g.fillRect(x, y + 2, 1, h);
      g.fillStyle = x < W / 2 ? human : ai; g.fillRect(x, y, 1, h);
    }
    if (!still) raf = requestAnimationFrame(frame);
  }
  return {
    run(on) {
      cancelAnimationFrame(raf);
      if (on) raf = requestAnimationFrame(frame);
    },
  };
})();

// ---------- routes ----------
function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1];
  const screen = SCREENS.includes(name) ? name : "home";
  const view = screen === "stats" ? "data" : screen; // Stats is the explorer's other view
  document.body.dataset.screen = view;
  $("data").dataset.tab = screen === "stats" ? "stats" : "clips";
  hero.run(screen === "home");
  if (view === "data") initExplorer().then(showExplorer, () => {});
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

route();
