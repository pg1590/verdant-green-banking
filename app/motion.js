/* Verdant motion controller. Native WAAPI + observers, no remote runtime.
 * The component only boots this singleton. All listeners/RAF live on the host
 * window; iframe replacement or a new build disposes and rebinds the controller.
 */
(() => {
  'use strict';
  let host, doc;
  try { host = window.parent; doc = host.document; } catch (_) { return; }
  const VERSION = 'verdant-motion-__BUILD__';
  if (!doc.querySelector || !host.IntersectionObserver || !host.Element.prototype.animate) return;
  // Rebind if Streamlit replaces the iframe, even with identical HTML. Callbacks
  // owned by a detached frame must never be reused by its replacement.
  if (host.__verdantMotion?.version === VERSION && host.__verdantMotion.owner === window) { host.__verdantMotion.scan(); return; }
  host.__verdantMotion?.destroy();

  const reduced = host.matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = host.matchMedia('(hover: hover) and (pointer: fine)');
  const ease = 'cubic-bezier(.22,1,.36,1)';
  const seen = new WeakSet();
  const queued = new Set();
  const active = new Map();
  const holds = new Map();
  const removers = [];
  let scanFrame = 0, depthFrame = 0, disposed = false, lastScroll = 0;
  let lastTime = 0, depthY = 0, depthTarget = 0, blur = 0, blurTarget = 0;
  let scene = null, scroller = null;

  function listen(target, name, handler, options) {
    target.addEventListener(name, handler, options);
    removers.push(() => target.removeEventListener(name, handler, options));
  }
  function play(el, frames, options = {}) {
    if (reduced.matches || !el?.isConnected || disposed) return;
    active.get(el)?.cancel();
    const animation = el.animate(frames, { duration: 620, easing: ease, ...options });
    active.set(el, animation);
    // No fill-forwards or permanent will-change: CSS owns the final state.
    animation.finished.catch(() => {}).then(() => {
      if (active.get(el) === animation) active.delete(el);
    });
  }
  // Leaf-level blocks only; an armed ancestor already carries its children.
  const REVEAL = [
    '.landing-principles > div', '.landing-products', '.st-key-landing-page [data-testid="stExpander"]', '.landing-footer',
    '.section-title', '.product-state', '.impact-tile', '.empty-state', '.product-header', '.app-footer',
    '[data-testid="stMain"] [data-testid="stMetric"]', '[data-testid="stMain"] [data-testid="stExpander"]',
    '[data-testid="stMain"] [data-testid="stPlotlyChart"]', '[data-testid="stMain"] [data-testid="stDataFrame"]',
    '[data-testid="stMain"] [data-testid="stAlert"]', '[data-testid="stMain"] [data-testid="stRadio"]',
    '[data-testid="stMain"] [data-testid="stSelectbox"]', '[data-testid="stMain"] [data-testid="stButton"]',
    '[data-testid="stMain"] [data-testid="stDownloadButton"]', '[data-testid="stMain"] [data-testid="stCaptionContainer"]',
    '[data-testid="stMain"] [data-testid="stMarkdownContainer"]', '[data-testid="stMain"] [data-testid="stJson"]',
    '[data-testid="stMain"] hr'
  ].join(', ');
  // A held opacity survives React className rewrites during Streamlit reruns.
  function hold(el) {
    el.classList.add('motion-pending');
    holds.set(el, el.animate([{ opacity: 0 }, { opacity: 0 }], { duration: 1, fill: 'forwards' }));
  }
  function release(el) {
    holds.get(el)?.cancel(); holds.delete(el);
    el.classList.remove('motion-pending');
  }
  function reveal(el, order = 0) {
    release(el);
    const column = el.closest('[data-testid="stColumn"]');
    const group = el.closest('.landing-principles, .product-states');
    const index = column ? [...column.parentElement.children].indexOf(column)
      : group ? [...group.children].indexOf(el) : 0;
    const heading = el.matches('.section-title');
    const small = el.matches('.landing-products, [data-testid="stCaptionContainer"], hr');
    const card = el.matches('[data-testid="stMetric"], .impact-tile, .product-state, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"]');
    play(el, [
      { opacity: 0, transform: heading ? 'translate3d(-18px,0,0)' : `translate3d(0,${small ? 10 : 30}px,0) scale(${card ? .97 : 1})`, filter: 'blur(4px)' },
      { opacity: 1, transform: 'translate3d(0,0,0) scale(1)', filter: 'blur(0px)' }
    ], { delay: Math.min(6, Math.max(index, order)) * 80, duration: heading ? 620 : 820, fill: 'backwards' });
  }
  const intersection = new host.IntersectionObserver(entries => {
    let order = 0;
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      intersection.unobserve(entry.target);
      queued.delete(entry.target);
      if (!reduced.matches) reveal(entry.target, order++);
      else release(entry.target);
    }
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  function scan() {
    if (disposed) return;
    for (const [el, animation] of active) {
      if (!el.isConnected) { animation.cancel(); active.delete(el); }
    }
    for (const el of queued) {
      if (!el.isConnected) { intersection.unobserve(el); queued.delete(el); holds.get(el)?.cancel(); holds.delete(el); }
    }
    if (!reduced.matches) {
      const hero = doc.querySelector('.hero');
      if (hero && !seen.has(hero)) {
        seen.add(hero);
        const navigating = active.has(hero.closest('.block-container'));
        play(hero, [
          { opacity: .6, transform: navigating ? 'none' : 'translate3d(8px,2px,0) scale(.995)' },
          { opacity: 1, transform: 'none' }
        ], { duration: 520 });
      }
      // Hero has its own CSS timeline; only supporting content uses reveals.
      for (const el of doc.querySelectorAll(REVEAL)) {
        if (seen.has(el)) continue;
        const bounds = el.getBoundingClientRect();
        // Hidden tab panels and collapsed content are retried once they render.
        if (!bounds.width && !bounds.height) continue;
        seen.add(el);
        if (el.parentElement?.closest(REVEAL) || el.closest('[role="tablist"], [data-testid="stSidebar"]')) continue;
        // Never hide visible content before JS starts (progressive enhancement).
        if (bounds.top <= host.innerHeight - 22) continue;
        hold(el);
        queued.add(el);
        intersection.observe(el);
      }
    }
    const nextScene = doc.querySelector('.landing-art');
    if (scene !== nextScene) {
      host.cancelAnimationFrame(depthFrame); depthFrame = 0;
      depthY = depthTarget = blur = blurTarget = 0;
      scene = nextScene;
    }
    const nextScroller = doc.querySelector('[data-testid="stMain"]');
    if (nextScroller !== scroller) {
      scroller?.removeEventListener('scroll', onScroll);
      scroller = nextScroller;
      scroller?.addEventListener('scroll', onScroll, { passive: true });
      lastScroll = scroller?.scrollTop || 0;
      lastTime = host.performance.now();
    }
  }
  function scheduleScan() {
    if (!scanFrame && !disposed) scanFrame = host.requestAnimationFrame(() => { scanFrame = 0; scan(); });
  }
  const mutation = new host.MutationObserver(scheduleScan);
  mutation.observe(doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body, { childList: true, subtree: true });

  function depthTick() {
    depthFrame = 0;
    if (!scene?.isConnected || reduced.matches || doc.hidden || disposed) return;
    depthY += (depthTarget - depthY) * .16;
    blur += (blurTarget - blur) * .18;
    blurTarget *= .72;
    scene.style.setProperty('--scene-y', `${depthY.toFixed(2)}px`);
    scene.style.setProperty('--scene-blur', `${blur.toFixed(2)}px`);
    if (Math.abs(depthTarget - depthY) > .08 || blur > .025) depthFrame = host.requestAnimationFrame(depthTick);
    else { scene.style.setProperty('--scene-blur', '0px'); }
  }
  function onScroll() {
    if (!scene || reduced.matches || disposed) return;
    const now = host.performance.now(), top = scroller?.scrollTop || 0;
    const velocity = Math.abs(top - lastScroll) / Math.max(16, now - lastTime);
    lastScroll = top; lastTime = now;
    const bounds = scene.getBoundingClientRect();
    if (bounds.bottom < 0 || bounds.top > host.innerHeight) return;
    // Bounded travel; velocity blur is restricted to the illustration.
    depthTarget = Math.min(14, Math.max(-14, top * .045));
    blurTarget = finePointer.matches ? Math.min(.65, velocity * .12) : 0;
    if (!depthFrame) depthFrame = host.requestAnimationFrame(depthTick);
  }
  function clearPointer() {
    for (const el of doc.querySelectorAll('.landing-art, .st-key-landing-customer button, .st-key-landing-business button')) {
      el.style.removeProperty('--pointer-x'); el.style.removeProperty('--pointer-y');
      el.style.removeProperty('--magnet-x'); el.style.removeProperty('--magnet-y');
    }
  }
  function onPointer(event) {
    if (reduced.matches || !finePointer.matches || event.pointerType === 'touch') return;
    const art = event.target.closest('.landing-art');
    const button = event.target.closest('.st-key-landing-customer button, .st-key-landing-business button');
    const el = art || button;
    if (!el) return;
    if (button?.disabled) {
      button.style.removeProperty('--magnet-x'); button.style.removeProperty('--magnet-y');
      return;
    }
    const box = el.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width - .5;
    const y = (event.clientY - box.top) / box.height - .5;
    el.style.setProperty(art ? '--pointer-x' : '--magnet-x', `${(x * (art ? 10 : 5)).toFixed(2)}px`);
    el.style.setProperty(art ? '--pointer-y' : '--magnet-y', `${(y * (art ? 8 : 4)).toFixed(2)}px`);
  }
  function onPointerOut(event) {
    const el = event.target.closest('.landing-art, .st-key-landing-customer button, .st-key-landing-business button');
    if (el && !el.contains(event.relatedTarget)) {
      for (const property of ['--pointer-x', '--pointer-y', '--magnet-x', '--magnet-y']) el.style.removeProperty(property);
    }
  }
  function onFocus(event) {
    const pending = event.target.closest('.motion-pending');
    if (pending) {
      intersection.unobserve(pending); queued.delete(pending);
      release(pending);
    }
  }
  function onIntroSkip(event) {
    const splash = doc.querySelector('.intro-splash:not(.intro-done)');
    if (!splash || (event.type === 'keydown' && !['Escape', 'Enter', ' '].includes(event.key))) return;
    if (event.type === 'click' && !event.target.closest('.intro-splash')) return;
    splash.classList.add('intro-done');
  }
  function onNavigation(event) {
    if (event.target.closest?.('[role="tab"], [data-testid="stExpander"] summary')) host.setTimeout(scheduleScan, 60);
    if (reduced.matches || event.defaultPrevented) return;
    const target = event.target.closest('.st-key-landing-customer button, .st-key-landing-business button, [data-testid="stSidebar"] [role="radiogroup"] label, .st-key-open-customer button');
    if (!target || target.disabled) return;
    if (target.matches('label') && target.querySelector('input:checked')) return;
    const page = doc.querySelector('[data-testid="stMain"] .block-container');
    const returning = target.textContent.trim() === 'Home';
    play(page, [
      { opacity: 1, transform: 'translate3d(0,0,0) scale(1)', filter: 'blur(0px)' },
      { opacity: .9, transform: `translate3d(${returning ? 5 : -5}px,-2px,0) scale(.997)`, filter: 'blur(.7px)', offset: .45 },
      { opacity: 1, transform: 'translate3d(0,0,0) scale(1)', filter: 'blur(0px)' }
    ], { duration: 360 });
    // Native clicks proceed immediately. Never delay a Streamlit callback.
  }
  function reset() {
    intersection.disconnect();
    for (const el of queued) { release(el); seen.delete(el); }
    queued.clear();
    for (const animation of active.values()) animation.cancel();
    active.clear();
    host.cancelAnimationFrame(depthFrame); depthFrame = 0;
    depthY = depthTarget = blur = blurTarget = 0;
    if (scene) { scene.style.removeProperty('--scene-y'); scene.style.removeProperty('--scene-blur'); }
    clearPointer();
  }
  function onPreference() { reset(); scan(); }
  function onVisibility() { if (doc.hidden) reset(); else scan(); }
  function destroy() {
    disposed = true; reset(); mutation.disconnect();
    host.cancelAnimationFrame(scanFrame);
    scroller?.removeEventListener('scroll', onScroll);
    removers.forEach(remove => remove());
    delete host.__verdantMotion;
  }
  listen(doc, 'pointermove', onPointer, { passive: true });
  listen(doc, 'pointerout', onPointerOut, { passive: true });
  listen(doc, 'click', onNavigation, true);
  listen(doc, 'focusin', onFocus);
  listen(doc, 'click', onIntroSkip);
  listen(doc, 'keydown', onIntroSkip);
  listen(doc, 'visibilitychange', onVisibility);
  listen(reduced, 'change', onPreference);
  listen(finePointer, 'change', clearPointer);
  listen(host, 'pagehide', event => { if (event.persisted) reset(); else destroy(); });
  listen(host, 'pageshow', event => { if (event.persisted) scan(); });
  host.__verdantMotion = {
    version: VERSION, owner: window, scan, destroy,
    // Useful for local QA without telemetry or customer data.
    status: () => ({ reduced: reduced.matches, pending: queued.size, active: active.size, frame: Boolean(depthFrame), disposed })
  };
  scan();
})();
