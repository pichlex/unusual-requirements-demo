import { HapticXYPad } from './xy-pad.js';

const $ = (selector, scope = document) => scope?.querySelector(selector);

function insertCard() {
  const grid = $('.component-grid');
  if (!grid) return null;

  const existing = $('#xy-card');
  if (existing) return existing;

  const card = document.createElement('article');
  card.id = 'xy-card';
  card.className = 'interactive-card component-card xy-card';
  card.innerHTML = `
    <div class="component-header xy-header">
      <div>
        <h3 class="component-title">Haptic field</h3>
        <p class="component-copy">Свободно веди точку по двум осям. Каждое пересечение линии сетки становится отдельным коротким откликом.</p>
      </div>
      <span class="component-number">07</span>
    </div>

    <div class="component-stage xy-stage">
      <div
        id="xy-surface"
        class="xy-surface"
        tabindex="0"
        role="application"
        aria-label="Двумерное тактильное поле. Используйте жест, мышь или стрелки клавиатуры."
      >
        <div class="xy-grid" aria-hidden="true"></div>
        <div id="xy-active-cell" class="xy-active-cell" aria-hidden="true"></div>
        <div class="xy-axis xy-axis-x" aria-hidden="true"></div>
        <div class="xy-axis xy-axis-y" aria-hidden="true"></div>
        <div id="xy-orb" class="xy-orb" aria-hidden="true">
          <span class="xy-orb-core"></span>
        </div>
        <div class="xy-readout" aria-hidden="true">
          <span><b id="xy-x">52</b><small>X</small></span>
          <span><b id="xy-y">54</b><small>Y</small></span>
        </div>
        <input
          id="xy-driver"
          class="xy-driver"
          type="checkbox"
          switch
          aria-label="Двумерное тактильное поле"
        >
      </div>

      <div class="xy-caption">
        <span>Веди пальцем или курсором</span>
        <span>Стрелки клавиатуры тоже работают</span>
      </div>
    </div>
  `;

  grid.append(card);
  requestAnimationFrame(() => card.classList.add('is-mounted'));

  const sectionDescription = $('.section-description', $('#components'));
  if (sectionDescription) {
    sectionDescription.textContent =
      'Вкладки, кнопка, переключатель, круговая крутилка и свободное двумерное поле — с единым характером движения.';
  }

  return card;
}

function replay(element, className) {
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}

function setup() {
  const card = insertCard();
  if (!card) return null;

  const surface = $('#xy-surface', card);
  const driver = $('#xy-driver', card);
  const orb = $('#xy-orb', card);
  const activeCell = $('#xy-active-cell', card);
  const xOutput = $('#xy-x', card);
  const yOutput = $('#xy-y', card);

  const columns = 10;
  const rows = 8;

  const pad = new HapticXYPad({
    surface,
    driver,
    columns,
    rows,
    x: 52,
    y: 46,
    onValue: ({ x, y, ratioX, ratioY, cellX, cellY, animate }) => {
      surface.style.setProperty('--xy-x', `${ratioX * 100}%`);
      surface.style.setProperty('--xy-y', `${ratioY * 100}%`);
      surface.style.setProperty('--xy-cell-x', String(cellX));
      surface.style.setProperty('--xy-cell-y', String(cellY));
      orb.style.left = `${ratioX * 100}%`;
      orb.style.top = `${ratioY * 100}%`;
      xOutput.textContent = String(Math.round(x));
      yOutput.textContent = String(Math.round(100 - y));
      surface.setAttribute(
        'aria-valuetext',
        `X ${Math.round(x)}, Y ${Math.round(100 - y)}`,
      );

      if (animate) {
        replay(orb, 'is-hit');
        replay(activeCell, 'is-hit');
      }
    },
    onTick: ({ axis, native }) => {
      surface.dataset.lastAxis = axis;
      surface.dataset.feedback = native ? 'native' : 'visual';
      replay(surface, 'is-grid-hit');
      replay(orb, 'is-hit');
      replay(activeCell, 'is-hit');
    },
  });

  window.hapticLab = {
    ...(window.hapticLab ?? {}),
    xyPad: pad,
  };

  return pad;
}

setup();
