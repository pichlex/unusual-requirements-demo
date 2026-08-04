import { bindHapticTap, bindNativeToggle } from './haptics.js';
import { AngularHapticDial, CrossPlatformRange } from './platform-haptics.js';

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

const hasTouch = navigator.maxTouchPoints > 0;
const hasFinePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
document.documentElement.classList.add(hasTouch ? 'touch-capable' : 'desktop-pointer');
if (hasFinePointer) document.documentElement.classList.add('fine-pointer');

function formatValue(value, digits = 0) {
  return Number(value).toFixed(digits);
}

function buildTicks(container, { min, max, step, majorEvery, mediumEvery }) {
  const values = [];
  for (let value = min; value <= max + 0.0001; value += step) {
    values.push(Number(value.toFixed(6)));
  }

  container.style.gridTemplateColumns = `repeat(${values.length}, minmax(0, 1fr))`;

  for (const value of values) {
    const tick = document.createElement('span');
    tick.className = 'range-tick';
    tick.dataset.value = String(value);

    if (majorEvery && Math.abs(value % majorEvery) < 0.0001) {
      tick.classList.add('is-major');
    } else if (mediumEvery && Math.abs(value % mediumEvery) < 0.0001) {
      tick.classList.add('is-medium');
    }

    container.append(tick);
  }

  return values;
}

function pulseTick(container, value, phase = 'primary') {
  const ticks = $$('.range-tick', container);
  let closest = null;
  let distance = Number.POSITIVE_INFINITY;

  for (const tick of ticks) {
    const nextDistance = Math.abs(Number(tick.dataset.value) - value);
    if (nextDistance < distance) {
      closest = tick;
      distance = nextDistance;
    }
  }

  if (!closest) return;

  closest.classList.remove('is-hit', 'is-echo');
  void closest.offsetWidth;
  closest.classList.add(phase === 'echo' ? 'is-echo' : 'is-hit');
}

function createTicker(element, initial = 'Ready') {
  const entries = [initial];

  return (message) => {
    entries.unshift(message);
    element.textContent = entries.slice(0, 3).join('  ·  ');
  };
}

function setupLinearDemo({
  id,
  step,
  majorEvery,
  mediumEvery,
  value = 50,
  accentOffset = 0,
  valueDigits = 0,
}) {
  const card = $(`#${id}-card`);
  const surface = $(`#${id}-surface`);
  const driver = $(`#${id}-driver`);
  const fill = $(`#${id}-fill`);
  const thumb = $(`#${id}-thumb`);
  const output = $(`#${id}-value`);
  const ticks = $(`#${id}-ticks`);
  const status = $(`#${id}-status`);
  const ticker = createTicker(
    status,
    hasTouch ? 'Touch, hold briefly, then drag' : 'Drag with mouse or trackpad',
  );

  buildTicks(ticks, {
    min: 0,
    max: 100,
    step,
    majorEvery,
    mediumEvery,
  });

  const render = ({ value: nextValue, ratio, animate }) => {
    const rect = surface.getBoundingClientRect();
    const inset = 16;
    const usable = Math.max(1, rect.width - inset * 2);
    const x = inset + usable * ratio;

    fill.style.width = `${usable * ratio}px`;
    thumb.style.left = `${x}px`;
    surface.style.setProperty('--active-x', `${x}px`);
    surface.style.setProperty('--progress', String(ratio));
    output.value = formatValue(nextValue, valueDigits);
    output.textContent = formatValue(nextValue, valueDigits);

    if (animate) {
      thumb.classList.remove('is-hit');
      void thumb.offsetWidth;
      thumb.classList.add('is-hit');
    }
  };

  const range = new CrossPlatformRange({
    surface,
    driver,
    min: 0,
    max: 100,
    step,
    value,
    trackInset: 16,
    accentOffset,
    isMajor: (tickValue) =>
      Boolean(majorEvery) && Math.abs(tickValue % majorEvery) < 0.0001,
    onValue: render,
    onTick: ({ value: tickValue, kind, phase, direction, native }) => {
      pulseTick(ticks, tickValue, phase);

      const arrow = direction > 0 ? '→' : '←';
      const source = native ? 'native' : 'visual';
      const label =
        phase === 'echo'
          ? `accent echo ${formatValue(tickValue, valueDigits)}`
          : `${kind} ${formatValue(tickValue, valueDigits)} ${arrow} · ${source}`;

      ticker(label);

      card.dispatchEvent(
        new CustomEvent('haptic-tick', {
          detail: { value: tickValue, kind, phase, direction, native },
        }),
      );
    },
    onDebug: ({ message }) => {
      card.dataset.debug = message;
    },
  });

  requestAnimationFrame(() => range.setValue(value));
  return range;
}

const coarseRange = setupLinearDemo({
  id: 'coarse',
  step: 10,
  majorEvery: 20,
  mediumEvery: 10,
  value: 50,
});

const denseRange = setupLinearDemo({
  id: 'dense',
  step: 2,
  majorEvery: 10,
  mediumEvery: 5,
  value: 42,
});

const intensityRange = setupLinearDemo({
  id: 'intensity',
  step: 5,
  majorEvery: 20,
  mediumEvery: 10,
  value: 35,
  accentOffset: 1.35,
});

function setupDial() {
  const surface = $('#dial-surface');
  const driver = $('#dial-driver');
  const output = $('#dial-value');
  const pointer = $('#dial-pointer');
  const halo = $('#dial-halo');
  const status = $('#dial-status');
  const ticker = createTicker(
    status,
    hasTouch ? 'Trace the arc with your finger' : 'Drag around the dial rim',
  );

  const dial = new AngularHapticDial({
    surface,
    driver,
    min: 0,
    max: 100,
    step: 5,
    value: 64,
    arcStart: -135,
    arcEnd: 135,
    isMajor: (value) => value % 25 === 0,
    onValue: ({ value, ratio, animate }) => {
      const angle = -135 + ratio * 270;
      output.textContent = String(Math.round(value));
      pointer.style.transform = `rotate(${angle}deg)`;
      halo.style.setProperty('--dial-progress', `${ratio * 270}deg`);
      halo.style.setProperty('--dial-angle', `${angle}deg`);

      if (animate) {
        halo.classList.remove('is-hit');
        void halo.offsetWidth;
        halo.classList.add('is-hit');
      }
    },
    onTick: ({ value, kind, direction, native }) => {
      ticker(
        `${kind} ${Math.round(value)} ${direction > 0 ? '↻' : '↺'} · ${native ? 'native' : 'visual'}`,
      );
    },
  });

  requestAnimationFrame(() => dial.setValue(64));
  return dial;
}

const dial = setupDial();

function setupTabs() {
  const root = $('#haptic-tabs');
  const indicator = $('.tabs-indicator', root);
  const items = $$('.haptic-tab', root);
  const panels = $$('.tab-panel', $('#tab-panels'));
  const status = $('#tabs-status');
  const ticker = createTicker(status, 'Tap any tab');

  const select = (index) => {
    root.style.setProperty('--tab-index', String(index));
    indicator.style.width = `${100 / items.length}%`;

    items.forEach((item, itemIndex) => {
      item.classList.toggle('is-active', itemIndex === index);
      item.setAttribute('aria-selected', String(itemIndex === index));
    });

    panels.forEach((panel, panelIndex) => {
      panel.classList.toggle('is-active', panelIndex === index);
    });

    ticker(`selected ${items[index].dataset.label}`);
  };

  items.forEach((item, index) => {
    const input = $('.haptic-tap-input', item);
    bindHapticTap(input, () => select(index));
  });

  select(0);
}

setupTabs();

function setupButton() {
  const button = $('#pulse-button');
  const input = $('.haptic-tap-input', button);
  const count = $('#pulse-count');
  const status = $('#button-status');
  const ticker = createTicker(status, 'Ready to launch');
  let launches = 0;

  const createParticle = () => {
    const particle = document.createElement('span');
    particle.className = 'button-particle';
    const angle = Math.random() * Math.PI * 2;
    const distance = 44 + Math.random() * 44;
    particle.style.setProperty('--particle-x', `${Math.cos(angle) * distance}px`);
    particle.style.setProperty('--particle-y', `${Math.sin(angle) * distance}px`);
    button.append(particle);
    particle.addEventListener('animationend', () => particle.remove());
  };

  bindHapticTap(input, () => {
    launches += 1;
    count.textContent = String(launches).padStart(2, '0');
    button.classList.remove('is-fired');
    void button.offsetWidth;
    button.classList.add('is-fired');

    for (let index = 0; index < 12; index += 1) createParticle();
    ticker(`pulse ${String(launches).padStart(2, '0')} fired`);
  });
}

setupButton();

function setupToggle() {
  const card = $('#toggle-card');
  const control = $('#ambient-toggle-control');
  const input = $('#ambient-toggle');
  const label = $('#toggle-state');
  const status = $('#toggle-status');
  const ticker = createTicker(status, 'Custom surface · native touch target');

  bindNativeToggle(input, ({ checked }) => {
    card.classList.toggle('is-enabled', checked);
    control.classList.toggle('is-on', checked);
    document.body.classList.toggle('ambient-enabled', checked);
    label.textContent = checked ? 'Atmosphere on' : 'Atmosphere off';
    ticker(checked ? 'enabled' : 'disabled');
  });
}

setupToggle();

function setupHeroHaptic() {
  const root = $('#hero-haptic');
  const input = $('.haptic-tap-input', root);
  const copy = $('.hero-haptic-copy', root);
  let count = 0;

  bindHapticTap(input, () => {
    count += 1;
    root.classList.remove('is-hit');
    void root.offsetWidth;
    root.classList.add('is-hit');
    copy.textContent = count === 1 ? 'You felt it.' : `${count} native ticks`;
  });
}

setupHeroHaptic();

function setupReveal() {
  const elements = $$('[data-reveal]');

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.12 },
  );

  elements.forEach((element) => observer.observe(element));
}

setupReveal();

function setupNav() {
  const nav = $('.site-nav');
  const update = () => nav.classList.toggle('is-scrolled', window.scrollY > 18);
  update();
  window.addEventListener('scroll', update, { passive: true });
}

setupNav();

function setupPointerGlow() {
  if (!hasFinePointer) return;

  const cards = $$('.interactive-card');
  for (const card of cards) {
    card.addEventListener('pointermove', (event) => {
      if (event.pointerType === 'touch') return;
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--pointer-x', `${event.clientX - rect.left}px`);
      card.style.setProperty('--pointer-y', `${event.clientY - rect.top}px`);
    });
  }
}

setupPointerGlow();

function setupScrollLinks() {
  $$('[data-scroll-target]').forEach((link) => {
    link.addEventListener('click', (event) => {
      const target = $(link.dataset.scrollTarget);
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

setupScrollLinks();

window.hapticLab = {
  coarseRange,
  denseRange,
  intensityRange,
  dial,
};
