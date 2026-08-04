import { HapticXYPad } from './xy-pad.js';

const $ = (selector, scope = document) => scope?.querySelector(selector);
const $$ = (selector, scope = document) => [...(scope?.querySelectorAll(selector) ?? [])];

function replaceText(selector, text, scope = document) {
  const element = $(selector, scope);
  if (element) element.textContent = text;
}

function insertXYCard() {
  const grid = $('.component-grid');
  if (!grid || $('#xy-card')) return;

  const card = document.createElement('article');
  card.id = 'xy-card';
  card.className = 'interactive-card component-card xy-card reveal';
  card.dataset.reveal = '';
  card.innerHTML = `
    <div class="component-header">
      <div>
        <h3 class="component-title">Поле</h3>
        <p class="component-copy">Свободное движение в двух измерениях. Сетка ощущается на каждом пересечении.</p>
      </div>
    </div>

    <div class="component-stage xy-stage">
      <div id="xy-surface" class="xy-surface" aria-label="Двумерное тактильное поле">
        <div class="xy-grid" aria-hidden="true"></div>
        <div id="xy-active-cell" class="xy-active-cell" aria-hidden="true"></div>
        <div class="xy-axis xy-axis-x" aria-hidden="true"></div>
        <div class="xy-axis xy-axis-y" aria-hidden="true"></div>
        <div id="xy-orb" class="xy-orb" aria-hidden="true">
          <span class="xy-orb-core"></span>
        </div>
        <div class="xy-readout" aria-hidden="true">
          <span><b id="xy-x">52</b> X</span>
          <span><b id="xy-y">46</b> Y</span>
        </div>
        <input id="xy-driver" class="haptic-driver xy-driver" type="checkbox" switch aria-label="Двумерное тактильное поле">
      </div>
      <p class="control-hint">Веди пальцем или курсором по полю.</p>
    </div>`;

  const tabsCard = $('.tabs-card', grid);
  grid.insertBefore(card, tabsCard ?? grid.firstChild);
}

function cleanMarkupBeforeBaseApp() {
  $('.nav-status')?.remove();
  $('.trust-strip')?.remove();
  $('.hero-footnote')?.remove();

  replaceText('.eyebrow', 'Тактильный интерфейс');
  const heroTitle = $('.hero-title');
  if (heroTitle) heroTitle.innerHTML = 'Интерфейс, который <span class="gradient-text">чувствуется.</span>';
  replaceText(
    '.hero-description',
    'Слайдеры, кнопки и контролы с настоящей тактильной отдачей на iPhone и полноценным управлением на компьютере.',
  );
  replaceText('.primary-cta', 'Попробовать');
  replaceText('.secondary-cta', 'Посмотреть API');
  replaceText('.device-topbar span:first-child', 'Демонстрация');
  replaceText('.device-live', 'Касание');
  replaceText('.device-kicker', 'Движение');
  const deviceTitle = $('.device-title');
  if (deviceTitle) deviceTitle.innerHTML = 'Двигай.<br>Чувствуй.';
  replaceText('.hero-haptic-title', 'Нажми');
  replaceText('.hero-haptic-copy', 'Один короткий системный отклик');

  const navLabels = ['Слайдеры', 'Элементы', 'Пакет'];
  $$('.nav-link').forEach((link, index) => {
    if (navLabels[index]) link.textContent = navLabels[index];
  });

  replaceText('.section-title', 'Три характера движения.', $('#sliders'));
  replaceText(
    '.section-description',
    'От крупных фиксированных шагов до плотной шкалы и акцентных делений.',
    $('#sliders'),
  );

  const sliderCards = [
    {
      id: 'coarse-card',
      title: 'Точный',
      description: 'Ровный шаг и заметные ориентиры для основных настроек.',
      unit: '',
    },
    {
      id: 'dense-card',
      title: 'Плотный',
      description: 'Больше делений для тонкой настройки и быстрых жестов.',
      unit: '',
    },
    {
      id: 'intensity-card',
      title: 'С акцентами',
      description: 'Обычные деления ощущаются легко, ключевые — заметнее.',
      unit: '',
    },
  ];

  sliderCards.forEach(({ id, title, description, unit }) => {
    const card = $(`#${id}`);
    if (!card) return;
    replaceText('.card-title', title, card);
    replaceText('.card-description', description, card);
    replaceText('.value-unit', unit, card);
  });

  const legend = $('.intensity-legend');
  if (legend) {
    legend.innerHTML = `
      <span class="legend-pill"><span class="legend-mark"></span>обычное деление</span>
      <span class="legend-pill"><span class="legend-mark is-accent"></span>акцентное деление</span>`;
  }

  replaceText('.section-title', 'Больше, чем слайдер.', $('#components'));
  replaceText(
    '.section-description',
    'Те же ощущения в навигации, действиях, переключателях и свободном движении.',
    $('#components'),
  );

  insertXYCard();

  replaceText('.package-title', 'Подключай как обычный компонент.');
  replaceText(
    '.package-copy',
    'Логика отделена от оформления: можно использовать готовые контролы или собрать собственный интерфейс поверх небольшого ES-модуля.',
  );
  replaceText('.site-footer .footer-inner > span:last-child', 'Haptic Lab · экспериментальные веб-контролы');
}

cleanMarkupBeforeBaseApp();

await import('../haptic-lab-v2/app.js');

function cleanMarkupAfterBaseApp() {
  replaceText(
    '.section-description',
    'На компьютере всё работает мышью и трекпадом. На iPhone движение дополнено системной тактильной сеткой.',
    $('#sliders'),
  );

  const componentCopy = [
    ['.tabs-card', 'Вкладки', 'Быстрое переключение между разделами с коротким откликом.'],
    ['.button-card', 'Кнопка', 'Прямое действие с аккуратной пружиной и вспышкой вокруг точки нажатия.'],
    ['#toggle-card', 'Переключатель', 'Полностью кастомный переключатель с мягким магнитным движением.'],
    ['#dial-card', 'Крутилка', 'Следует за реальным углом пальца вокруг центра.'],
  ];

  componentCopy.forEach(([selector, title, copy]) => {
    const card = $(selector);
    if (!card) return;
    replaceText('.component-title', title, card);
    replaceText('.component-copy', copy, card);
  });

  replaceText('.pulse-counter span', 'Нажатий');
  replaceText('#toggle-state', 'Выключено');
  replaceText('.dial-unit', 'уровень');

  const tabNames = ['Обзор', 'Движение', 'Отклик'];
  $$('.haptic-tab').forEach((tab, index) => {
    const label = $('.haptic-tab-icon', tab);
    const textNode = label
      ? [...label.childNodes].find(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim())
      : null;
    if (textNode && tabNames[index]) textNode.textContent = ` ${tabNames[index]}`;
    if (tabNames[index]) tab.dataset.label = tabNames[index];
  });

  const toggleControl = $('#ambient-toggle-control');
  const toggleState = $('#toggle-state');
  if (toggleControl && toggleState) {
    const syncToggleCopy = () => {
      const next = toggleControl.getAttribute('aria-checked') === 'true'
        ? 'Включено'
        : 'Выключено';
      if (toggleState.textContent !== next) toggleState.textContent = next;
    };
    new MutationObserver(syncToggleCopy).observe(toggleControl, {
      attributes: true,
      attributeFilter: ['aria-checked'],
    });
    syncToggleCopy();
  }

  const heroFeedback = $('.hero-haptic-copy');
  if (heroFeedback) {
    const syncHeroCopy = () => {
      if (heroFeedback.textContent !== 'Отклик получен') {
        heroFeedback.textContent = 'Отклик получен';
      }
    };
    new MutationObserver(syncHeroCopy).observe(heroFeedback, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }

  const dialStage = $('.dial-layout');
  if (dialStage && !$('.dial-hint', dialStage)) {
    const hint = document.createElement('p');
    hint.className = 'control-hint dial-hint';
    hint.textContent = 'Веди пальцем по окружности.';
    dialStage.append(hint);
  }

  const packageLinks = $$('.package-card .hero-actions a');
  if (packageLinks[0]) {
    packageLinks[0].textContent = 'Открыть модуль';
    packageLinks[0].href = '../haptic-lab-v2/haptics.js';
  }
  if (packageLinks[1]) {
    packageLinks[1].textContent = 'Документация';
    packageLinks[1].href = '../haptic-lab/README.md';
  }

  const tabPanels = $$('.tab-panel');
  const panelCopy = [
    ['Один язык движения', 'Все контролы реагируют быстро и одинаково, даже если выглядят по-разному.'],
    ['Без лишнего шума', 'Анимация подчёркивает действие и сразу возвращает внимание к содержанию.'],
    ['Физический отклик', 'На iPhone короткий импульс создаёт сам браузерный системный контрол.'],
  ];
  tabPanels.forEach((panel, index) => {
    const [title, copy] = panelCopy[index] ?? [];
    if (!title) return;
    replaceText('h4', title, panel);
    replaceText('p', copy, panel);
  });
}

cleanMarkupAfterBaseApp();

function setupXYPad() {
  const surface = $('#xy-surface');
  const driver = $('#xy-driver');
  const orb = $('#xy-orb');
  const activeCell = $('#xy-active-cell');
  const xOutput = $('#xy-x');
  const yOutput = $('#xy-y');

  if (!surface || !driver) return null;

  const columns = 10;
  const rows = 8;

  const pad = new HapticXYPad({
    surface,
    driver,
    columns,
    rows,
    x: 52,
    y: 46,
    onValue: ({ x, y, ratioX, ratioY, animate }) => {
      surface.style.setProperty('--xy-x', `${ratioX * 100}%`);
      surface.style.setProperty('--xy-y', `${ratioY * 100}%`);
      surface.style.setProperty('--xy-cell-x', String(Math.min(columns - 1, Math.floor(ratioX * columns))));
      surface.style.setProperty('--xy-cell-y', String(Math.min(rows - 1, Math.floor(ratioY * rows))));
      orb.style.left = `${ratioX * 100}%`;
      orb.style.top = `${ratioY * 100}%`;
      xOutput.textContent = String(Math.round(x));
      yOutput.textContent = String(Math.round(100 - y));

      if (animate) {
        surface.classList.remove('is-grid-hit');
        activeCell.classList.remove('is-hit');
        void surface.offsetWidth;
        surface.classList.add('is-grid-hit');
        activeCell.classList.add('is-hit');
      }
    },
    onTick: () => {
      surface.classList.remove('is-grid-hit');
      activeCell.classList.remove('is-hit');
      void surface.offsetWidth;
      surface.classList.add('is-grid-hit');
      activeCell.classList.add('is-hit');
    },
  });

  window.hapticLab = {
    ...(window.hapticLab ?? {}),
    xyPad: pad,
  };

  return pad;
}

setupXYPad();
