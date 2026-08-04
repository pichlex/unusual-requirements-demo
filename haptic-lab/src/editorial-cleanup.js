const $ = (selector, scope = document) => scope?.querySelector(selector);
const $$ = (selector, scope = document) => [...(scope?.querySelectorAll(selector) ?? [])];

function setText(selector, text, scope = document) {
  const element = $(selector, scope);
  if (element) element.textContent = text;
}

function setHTML(selector, html, scope = document) {
  const element = $(selector, scope);
  if (element) element.innerHTML = html;
}

function remove(selector, scope = document) {
  $$(selector, scope).forEach((element) => element.remove());
}

/* Remove all runtime/debug presentation. Existing controllers may keep
   references to detached nodes, so their internal behaviour stays intact. */
remove([
  '.nav-status',
  '.trust-strip',
  '.hero-footnote',
  '.section-kicker',
  '.card-eyebrow',
  '.component-number',
  '.card-footer-row',
  '.haptic-status',
  '.card-meta',
  '.toggle-state-row',
  '.dial-info',
  '.device-live',
].join(','));

/* Navigation and hero. */
const navLabels = ['Слайдеры', 'Элементы', 'Код'];
$$('.nav-link').forEach((link, index) => {
  if (navLabels[index]) link.textContent = navLabels[index];
});

setText('.eyebrow', 'Тактильные веб-интерфейсы');
setHTML('.hero-title', 'Управление, которое <span class="gradient-text">чувствуется.</span>');
setText(
  '.hero-description',
  'Привычные элементы управления с тактильной отдачей на iPhone и полноценной работой на компьютере.',
);
setText('.primary-cta', 'Попробовать');
setText('.secondary-cta', 'Посмотреть код');
setText('.device-topbar span:first-child', 'Демонстрация');
setText('.device-kicker', 'Отклик');
setHTML('.device-title', 'Двигай.<br>Чувствуй.');
setText('.hero-haptic-title', 'Нажми');
setText('.hero-haptic-copy', 'Короткий системный отклик');

/* Slider section. */
const slidersSection = $('#sliders');
setText('.section-title', 'Три характера движения.', slidersSection);
setText(
  '.section-description',
  'Крупные шаги, плотная шкала и заметные акценты — без служебных подписей и потока событий.',
  slidersSection,
);

const sliderCopy = [
  ['#coarse-card', 'Точный', 'Ровный шаг и ясные ориентиры для основных настроек.'],
  ['#dense-card', 'Плотный', 'Больше делений для тонкой настройки и быстрых жестов.'],
  ['#intensity-card', 'С акцентами', 'Обычные деления ощущаются легко, ключевые — заметнее.'],
];

sliderCopy.forEach(([selector, title, description]) => {
  const card = $(selector);
  if (!card) return;
  setText('.card-title', title, card);
  setText('.card-description', description, card);
  setText('.value-unit', '', card);
});

const legend = $('.intensity-legend');
if (legend) {
  legend.innerHTML = `
    <span class="legend-pill"><span class="legend-mark"></span>обычное деление</span>
    <span class="legend-pill"><span class="legend-mark is-accent"></span>акцентное деление</span>`;
}

/* Component section. */
const componentsSection = $('#components');
setText('.section-title', 'Больше, чем слайдер.', componentsSection);
setText(
  '.section-description',
  'Вкладки, кнопка, переключатель, крутилка и свободное двумерное поле.',
  componentsSection,
);

const components = [
  ['.tabs-card', 'Вкладки', 'Быстрое переключение между разделами с коротким откликом.'],
  ['.button-card', 'Кнопка', 'Прямое действие с аккуратной пружиной вокруг точки нажатия.'],
  ['#toggle-card', 'Переключатель', 'Мягкое магнитное движение и спокойная смена состояния.'],
  ['#dial-card', 'Крутилка', 'Следует за движением пальца или курсора по окружности.'],
  ['#xy-card', 'Поле', 'Свободно двигай точку по двум осям и ощущай линии сетки.'],
];

components.forEach(([selector, title, description]) => {
  const card = $(selector);
  if (!card) return;
  setText('.component-title', title, card);
  setText('.component-copy', description, card);
});

setText('.pulse-counter span', 'Нажатий');
setText('.dial-unit', 'уровень');

const xyCaption = $('.xy-caption');
if (xyCaption) {
  xyCaption.innerHTML = '<span>Веди пальцем, мышью или трекпадом</span>';
}

const tabLabels = ['Обзор', 'Движение', 'Отклик'];
$$('.haptic-tab').forEach((tab, index) => {
  const label = $('.haptic-tab-icon', tab);
  if (!label || !tabLabels[index]) return;
  const textNode = [...label.childNodes].find(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
  );
  if (textNode) textNode.textContent = ` ${tabLabels[index]}`;
  tab.dataset.label = tabLabels[index];
});

const panelCopy = [
  ['Один язык движения', 'Все контролы реагируют быстро и последовательно.'],
  ['Без лишнего шума', 'Анимация подчёркивает действие и сразу отступает.'],
  ['Физический отклик', 'На iPhone короткий импульс создаёт сам системный контрол.'],
];
$$('.tab-panel').forEach((panel, index) => {
  const copy = panelCopy[index];
  if (!copy) return;
  setText('h4', copy[0], panel);
  setText('p', copy[1], panel);
});

/* Keep copy human after the original demo controllers update their labels. */
const toggleState = $('#toggle-state');
if (toggleState) {
  const syncToggle = () => {
    const input = $('#ambient-toggle');
    const text = input?.checked ? 'Включено' : 'Выключено';
    if (toggleState.textContent !== text) toggleState.textContent = text;
  };
  new MutationObserver(syncToggle).observe(toggleState, {
    childList: true,
    characterData: true,
    subtree: true,
  });
  $('#ambient-toggle')?.addEventListener('change', syncToggle);
  syncToggle();
}

const heroFeedback = $('.hero-haptic-copy');
if (heroFeedback) {
  let hasInteracted = false;
  $('#hero-haptic input')?.addEventListener('change', () => {
    hasInteracted = true;
    heroFeedback.textContent = 'Отклик получен';
  });

  new MutationObserver(() => {
    const desired = hasInteracted ? 'Отклик получен' : 'Короткий системный отклик';
    if (heroFeedback.textContent !== desired) heroFeedback.textContent = desired;
  }).observe(heroFeedback, {
    childList: true,
    characterData: true,
    subtree: true,
  });
}

/* Package and footer. */
const packageSection = $('#package');
setText('.package-title', 'Подключается как обычный компонент.', packageSection);
setText(
  '.package-copy',
  'Логика отделена от оформления: бери готовые контролы или собирай собственные поверхности поверх небольшого ES-модуля.',
  packageSection,
);
setText('.site-footer .footer-inner > span:last-child', 'Haptic Lab · веб-контролы');
