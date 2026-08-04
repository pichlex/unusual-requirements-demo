import {
HapticRange,
bindHapticTap,
bindHapticToggle,
} from './haptics.js';
const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
function clamp(value, minimum, maximum) {
return Math.max(minimum, Math.min(maximum, value));
}
function formatValue(value, digits = 0) {
return Number(value).toFixed(digits);
}
function upgradeLegacyMarkup() {
const sliderDescription = $('.section-description', $('#sliders'));
if (sliderDescription) {
sliderDescription.textContent =
'На Mac двигай мышью или трекпадом. На iPhone зажми control на долю секунды и веди пальцем: tactile-пороги формируются настоящим Safari switch.';
}
const toggleCard = $('#toggle-card');
const oldToggle = $('#ambient-toggle');
if (toggleCard && oldToggle && !$('#ambient-toggle-control')) {
$('.component-title', toggleCard).textContent = 'Atmosphere toggle';
$('.component-copy', toggleCard).textContent =
'Полностью кастомная control surface с магнитным бегунком и скрытым native touch-layer на iPhone.';
const control = document.createElement('div');
control.id = 'ambient-toggle-control';
control.className = 'custom-toggle-control';
control.setAttribute('role', 'switch');
control.setAttribute('aria-checked', 'false');
control.setAttribute('tabindex', '0');
control.setAttribute('aria-label', 'Переключить атмосферное освещение');
control.innerHTML = `
<span class="custom-toggle-track" aria-hidden="true">
<span class="custom-toggle-icons"><span>✦</span><span>☾</span></span>
<span class="custom-toggle-thumb"><span class="custom-toggle-spark"></span></span>
</span>`;
oldToggle.className = 'haptic-tap-input custom-toggle-driver';
oldToggle.removeAttribute('style');
oldToggle.parentNode.replaceChild(control, oldToggle);
control.append(oldToggle);
}
const dialCard = $('#dial-card');
if (dialCard) {
$('.component-copy', dialCard).textContent =
'Настоящее круговое управление: значение вычисляется из угла пальца относительно центра, а не из горизонтального смещения.';
const stats = $$('.dial-stat', dialCard);
if (stats[0]) {
$('b', stats[0]).textContent = 'Circular gesture';
$('span', stats[0]).textContent =
'Веди пальцем по дуге вокруг центра — указатель следует за реальным углом.';
}
}
}
upgradeLegacyMarkup();
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
} else if (
mediumEvery &&
Math.abs(value % mediumEvery) < 0.0001
) {
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
matchMedia('(hover: hover) and (pointer: fine)').matches
? 'Drag with mouse or trackpad'
: 'Touch, hold briefly, then drag',
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
const range = new HapticRange({
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
onTick: ({ value: tickValue, kind, phase, direction }) => {
pulseTick(ticks, tickValue, phase);
const arrow = direction > 0 ? '→' : '←';
const label =
phase === 'echo'
? `accent echo ${formatValue(tickValue, valueDigits)}`
: `${kind} ${formatValue(tickValue, valueDigits)} ${arrow}`;
ticker(label);
card.dispatchEvent(
new CustomEvent('haptic-tick', {
detail: { value: tickValue, kind, phase, direction },
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
const ticker = createTicker(status, 'Drag around the circular track');
const minimumAngle = -135;
const maximumAngle = 135;
const sweep = maximumAngle - minimumAngle;
let interactionRadius = surface.clientWidth * 0.4;
const valueToAngle = (value) =>
minimumAngle + clamp(value / 100, 0, 1) * sweep;
const pointToValue = ({ clientX, clientY, surface, currentValue }) => {
const rect = surface.getBoundingClientRect();
const centerX = rect.left + rect.width / 2;
const centerY = rect.top + rect.height / 2;
const dx = clientX - centerX;
const dy = clientY - centerY;
interactionRadius = clamp(
Math.hypot(dx, dy),
rect.width * 0.27,
rect.width * 0.49,
);
let angle = Math.atan2(dx, -dy) * 180 / Math.PI;
if (angle > maximumAngle || angle < minimumAngle) {
angle = currentValue >= 50 ? maximumAngle : minimumAngle;
}
return (angle - minimumAngle) / sweep * 100;
};
const xForValue = ({ value, surface }) => {
const angle = valueToAngle(value) * Math.PI / 180;
return surface.clientWidth / 2 + interactionRadius * Math.sin(angle);
};
const geometryDirectionForTarget = ({ tick, valueDirection }) => {
const angle = valueToAngle(tick) * Math.PI / 180;
const horizontalDerivative = Math.cos(angle) * valueDirection;
return Math.abs(horizontalDerivative) < 0.04
? valueDirection
: Math.sign(horizontalDerivative);
};
const dial = new HapticRange({
surface,
driver,
min: 0,
max: 100,
step: 5,
value: 64,
trackInset: 0,
driverHeight: 260,
pointToValue,
xForValue,
geometryDirectionForTarget,
isMajor: (value) => value % 25 === 0,
onValue: ({ value, ratio, animate }) => {
const angle = minimumAngle + ratio * sweep;
output.textContent = String(Math.round(value));
pointer.style.transform = `rotate(${angle}deg)`;
halo.style.setProperty('--dial-progress', `${ratio * sweep}deg`);
halo.style.setProperty('--dial-angle', `${angle}deg`);
if (animate) {
halo.classList.remove('is-hit');
void halo.offsetWidth;
halo.classList.add('is-hit');
}
},
onTick: ({ value, kind, direction, phase }) => {
const suffix = phase === 'desktop' ? 'visual' : 'native';
ticker(`${kind} ${Math.round(value)} ${direction > 0 ? '↻' : '↺'} · ${suffix}`);
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
const ticker = createTicker(status, 'Custom surface · native touch layer');
bindHapticToggle({
control,
input,
initial: false,
onChange: ({ checked, source }) => {
card.classList.toggle('is-enabled', checked);
document.body.classList.toggle('ambient-enabled', checked);
label.textContent = checked ? 'Atmosphere on' : 'Atmosphere off';
ticker(`${checked ? 'enabled' : 'disabled'} · ${source}`);
},
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
if (!matchMedia('(hover: hover) and (pointer: fine)').matches) return;
const cards = $$('.interactive-card');
for (const card of cards) {
card.addEventListener('pointermove', (event) => {
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
