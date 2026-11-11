// Pixel icons: one string per row; "#" = currentColor, other letters = palette.
const ICONS = {
  home: ["....#....", "...###...", "..#####..", ".#######.", "#########", ".##...##.", ".##...##.", ".##...##.", ".#######."],
  wave: bars([2, 5, 8, 4, 11, 7, 3, 9, 6, 3, 1]),
  wave2: bars([2, 5, 8, 4, 11, 7, 3, 9, 6, 3, 1], "hhhhhhaaaaa"),
  chart: ["..h......", "..h...a..", ".hh...a..", ".hhh..aa.", "hhhh.aaa.", "hhhh.aaaa", "#########"],
  chip: ["..#.#.#..", ".#######.", "##.....##", ".#.###.#.", "##.###.##", ".#.###.#.", "##.....##", ".#######.", "..#.#.#.."],
  mic: ["...###...", "..#####..", "..#####..", "..#####..", "..#####..", "#.#####.#", "#..###..#", ".#.....#.", "..#####..", "....#....", "..#####.."],
  ext: ["..#####", ".....##", "....###", "...##.#", "..##..#", ".##....", "##....."],
  dice: ["#########", "#.......#", "#.#...#.#", "#.......#", "#...#...#", "#.......#", "#.#...#.#", "#.......#", "#########"],
};
const PALETTE = { h: "var(--human)", a: "var(--ai)" };

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
// Draw each [data-pix] element under root.
export function fillPix(root = document) {
  root.querySelectorAll("[data-pix]").forEach((el) => { el.innerHTML = pix(ICONS[el.dataset.pix]); });
}
