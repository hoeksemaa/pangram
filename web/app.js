// The app shell: home screen, four tabs (hash routes), pixel icons, and the hero waveform.
// Data and Stats are two views of the explorer (explorer.js). Model and Try it: model.js.
import { initExplorer, showExplorer } from "./explorer.js";
import { fillPix } from "./icons.js";
import { leaveTry, showModel, showTry } from "./model.js";

const $ = (id) => document.getElementById(id);
const SCREENS = ["home", "data", "stats", "model", "try"];
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

fillPix();

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
  if (screen === "model") showModel();
  if (screen === "try") showTry(); else leaveTry();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

route();
