const DEFAULTS = {
min: 0,
max: 100,
step: 10,
value: 50,
trackInset: 14,
driverHeight: 72,
firstArmMs: 228,
parkWidth: 1200,
parkHeight: 100,
parkLocalLeft: 1,
startOffsetProportion: 0.4,
directionEpsilon: 0.65,
rearmDelayMs: 0,
accentOffset: 0,
pointToValue: null,
xForValue: null,
geometryDirectionForTarget: null,
isMajor: () => false,
onValue: () => {},
onTick: () => {},
onDebug: () => {},
};
function clamp(value, minimum, maximum) {
return Math.max(minimum, Math.min(maximum, value));
}
function touchById(touchList, identifier) {
for (const touch of touchList) {
if (touch.identifier === identifier) return touch;
}
return null;
}
function nowLabel() {
return new Date().toLocaleTimeString('ru-RU', { hour12: false });
}
export class HapticRange {
constructor(options) {
this.options = { ...DEFAULTS, ...options };
this.surface = this.options.surface;
this.driver = this.options.driver;
if (!(this.surface instanceof HTMLElement)) {
throw new TypeError('HapticRange requires an HTMLElement surface.');
}
if (!(this.driver instanceof HTMLInputElement)) {
throw new TypeError('HapticRange requires an input driver.');
}
this.driver.type = 'checkbox';
this.driver.setAttribute('switch', '');
this.value = clamp(
Number(this.options.value),
this.options.min,
this.options.max,
);
this.activePointerId = null;
this.activeTouchId = null;
this.desktopPointer = false;
this.gestureStartedAt = 0;
this.armTimer = null;
this.rearmTimer = null;
this.startClientX = 0;
this.startLocalFull = 0;
this.currentValue = this.value;
this.previousPointerValue = this.value;
this.previousTouchValue = this.value;
this.movementDirection = 0;
this.parkingApplied = false;
this.storedStartLocal = null;
this.firstArmed = false;
this.hasTicked = false;
this.visualStateOn = false;
this.targetTick = null;
this.targetMode = 'grid';
this.echoBaseTick = null;
this.pointerMoves = 0;
this.touchMoves = 0;
this.lastDebugAt = 0;
this.abortController = new AbortController();
this.signal = this.abortController.signal;
this.finePointerQuery = window.matchMedia('(hover: hover) and (pointer: fine)');
this.syncPointerMode();
this.bindEvents();
this.resetDriverFull();
this.emitValue(this.value, false);
}
bindEvents() {
window.addEventListener(
'pointerdown',
(event) => this.handlePointerDown(event),
{ capture: true, signal: this.signal },
);
window.addEventListener(
'pointermove',
(event) => this.handlePointerMove(event),
{ capture: true, signal: this.signal },
);
window.addEventListener(
'pointerup',
(event) => this.finishPointer(event),
{ capture: true, signal: this.signal },
);
window.addEventListener(
'pointercancel',
(event) => this.finishPointer(event),
{ capture: true, signal: this.signal },
);
this.surface.addEventListener(
'touchstart',
(event) => this.handleTouchStart(event),
{ capture: true, passive: true, signal: this.signal },
);
this.surface.addEventListener(
'touchmove',
(event) => this.handleTouchMove(event),
{ capture: true, passive: true, signal: this.signal },
);
this.surface.addEventListener(
'touchend',
(event) => this.finishTouch(event),
{ capture: true, passive: true, signal: this.signal },
);
this.surface.addEventListener(
'touchcancel',
(event) => this.finishTouch(event),
{ capture: true, passive: true, signal: this.signal },
);
this.finePointerQuery.addEventListener(
'change',
() => this.syncPointerMode(),
{ signal: this.signal },
);
window.addEventListener(
'resize',
() => this.emitValue(this.currentValue, false),
{ signal: this.signal },
);
}
syncPointerMode() {
const fine = this.finePointerQuery.matches;
this.surface.classList.toggle('is-fine-pointer', fine);
this.driver.style.pointerEvents = fine ? 'none' : 'auto';
this.surface.style.cursor = fine ? 'grab' : '';
}
debug(message) {
this.options.onDebug({
time: nowLabel(),
message,
value: this.currentValue,
target: this.targetTick,
direction: this.movementDirection,
});
}
metrics() {
const rect = this.surface.getBoundingClientRect();
const left = this.options.trackInset;
const right = rect.width - this.options.trackInset;
return {
rect,
left,
right,
width: Math.max(1, right - left),
};
}
valueFromPoint(clientX, clientY) {
if (typeof this.options.pointToValue === 'function') {
const mapped = this.options.pointToValue({
clientX,
clientY,
surface: this.surface,
currentValue: this.currentValue,
min: this.options.min,
max: this.options.max,
range: this,
});
return clamp(Number(mapped), this.options.min, this.options.max);
}
const { rect, width } = this.metrics();
const ratio = clamp(
(clientX - rect.left - this.options.trackInset) / width,
0,
1,
);
return this.options.min + ratio * (this.options.max - this.options.min);
}
xFromValue(value) {
if (typeof this.options.xForValue === 'function') {
const mapped = this.options.xForValue({
value,
surface: this.surface,
currentValue: this.currentValue,
min: this.options.min,
max: this.options.max,
range: this,
});
return Number(mapped);
}
const { left, width } = this.metrics();
const ratio =
(clamp(value, this.options.min, this.options.max) - this.options.min) /
(this.options.max - this.options.min);
return left + width * ratio;
}
geometryDirectionForTarget(tick, valueDirection) {
if (typeof this.options.geometryDirectionForTarget === 'function') {
const direction = Number(
this.options.geometryDirectionForTarget({
tick,
valueDirection,
surface: this.surface,
currentValue: this.currentValue,
min: this.options.min,
max: this.options.max,
range: this,
}),
);
if (direction !== 0 && Number.isFinite(direction)) {
return Math.sign(direction);
}
}
return Math.sign(valueDirection) || 1;
}
emitValue(value, animate) {
this.currentValue = clamp(value, this.options.min, this.options.max);
this.options.onValue({
value: this.currentValue,
animate,
ratio:
(this.currentValue - this.options.min) /
(this.options.max - this.options.min),
});
}
nextGridTick(value, direction) {
const { min, max, step } = this.options;
if (direction > 0) {
const index = Math.floor((value - min + 0.0001) / step) + 1;
const next = min + index * step;
return next < max ? next : null;
}
if (direction < 0) {
const index = Math.ceil((value - min - 0.0001) / step) - 1;
const next = min + index * step;
return next > min ? next : null;
}
return null;
}
crossed(fromValue, toValue, tick, direction) {
if (tick === null) return false;
if (direction > 0) return fromValue < tick && toValue >= tick;
if (direction < 0) return fromValue > tick && toValue <= tick;
return false;
}
resetDriverFull() {
clearTimeout(this.rearmTimer);
Object.assign(this.driver.style, {
left: '0px',
right: 'auto',
width: '100%',
height: `${this.options.driverHeight}px`,
direction: 'ltr',
});
void this.driver.offsetWidth;
}
applyParking(geometryDirection) {
const rect = this.surface.getBoundingClientRect();
const startX = this.startClientX - rect.left;
const parkLocalRight = this.options.parkWidth - 1;
if (geometryDirection > 0) {
this.driver.style.direction = 'ltr';
this.driver.style.left = `${startX - parkLocalRight}px`;
this.storedStartLocal = parkLocalRight;
} else {
this.driver.style.direction = 'rtl';
this.driver.style.left = `${startX - this.options.parkLocalLeft}px`;
this.storedStartLocal = this.options.parkLocalLeft;
}
Object.assign(this.driver.style, {
right: 'auto',
width: `${this.options.parkWidth}px`,
height: `${this.options.parkHeight}px`,
});
this.parkingApplied = true;
void this.driver.offsetWidth;
this.debug(
`park ${geometryDirection > 0 ? 'right' : 'left'}; local=${this.storedStartLocal}`,
);
}
effectiveStoredStartLocal() {
return this.parkingApplied ? this.storedStartLocal : this.startLocalFull;
}
applyFirstTargetGeometry(tick, geometryDirection) {
if (tick === null || geometryDirection === 0) return;
const targetX = this.xFromValue(tick);
const storedStart = this.effectiveStoredStartLocal();
const width = this.options.parkWidth;
if (geometryDirection > 0) {
const requiredHeight = Math.max(20, width - storedStart + 12);
const changePosition =
storedStart + this.options.startOffsetProportion * width;
Object.assign(this.driver.style, {
direction: 'ltr',
left: `${targetX - changePosition}px`,
right: 'auto',
width: `${width}px`,
height: `${requiredHeight}px`,
});
} else {
const requiredHeight = Math.max(
20,
Math.min(120, (width - storedStart) / 2),
);
const changePosition =
storedStart - this.options.startOffsetProportion * width;
Object.assign(this.driver.style, {
direction: 'rtl',
left: `${targetX - changePosition}px`,
right: 'auto',
width: `${width}px`,
height: `${requiredHeight}px`,
});
}
void this.driver.offsetWidth;
this.debug(
`first target=${tick}; geometry=${geometryDirection > 0 ? 'right' : 'left'}; stored=${Math.round(storedStart)}`,
);
}
rtlForSafeApproach(geometryDirection, stateOn) {
if (geometryDirection > 0) return stateOn;
return !stateOn;
}
applyNormalTargetGeometry(tick, geometryDirection) {
if (tick === null || geometryDirection === 0) return;
const { left, right } = this.metrics();
const tickX = this.xFromValue(tick);
const useRTL = this.rtlForSafeApproach(
geometryDirection,
this.visualStateOn,
);
this.driver.style.direction = useRTL ? 'rtl' : 'ltr';
if (geometryDirection > 0) {
Object.assign(this.driver.style, {
left: `${left}px`,
right: 'auto',
width: `${Math.max(
this.options.driverHeight + 2,
2 * (tickX - left),
)}px`,
});
} else {
Object.assign(this.driver.style, {
left: 'auto',
right: `${this.surface.clientWidth - right}px`,
width: `${Math.max(
this.options.driverHeight + 2,
2 * (right - tickX),
)}px`,
});
}
this.driver.style.height = `${this.options.driverHeight}px`;
void this.driver.offsetWidth;
this.debug(
`target=${tick}; geometry=${geometryDirection > 0 ? 'right' : 'left'}; state=${this.visualStateOn ? 'on' : 'off'}; ${useRTL ? 'rtl' : 'ltr'}`,
);
}
scheduleNormalTarget(tick, valueDirection, mode = 'grid', echoBase = null) {
clearTimeout(this.rearmTimer);
this.rearmTimer = setTimeout(() => {
this.targetTick = tick;
this.targetMode = mode;
this.echoBaseTick = echoBase;
if (tick !== null) {
const geometryDirection = this.geometryDirectionForTarget(
tick,
valueDirection,
);
this.applyNormalTargetGeometry(tick, geometryDirection);
}
}, this.options.rearmDelayMs);
}
armFirstNow() {
if (
this.activeTouchId === null ||
this.movementDirection === 0 ||
this.firstArmed
) {
return;
}
this.firstArmed = true;
this.targetTick = this.nextGridTick(
this.currentValue,
this.movementDirection,
);
this.targetMode = 'grid';
this.echoBaseTick = null;
const geometryDirection = this.geometryDirectionForTarget(
this.targetTick,
this.movementDirection,
);
this.applyFirstTargetGeometry(this.targetTick, geometryDirection);
this.debug(`native ready; first=${this.targetTick}`);
}
setDirectionBeforeFirst(direction) {
this.movementDirection = direction;
this.targetTick = this.nextGridTick(this.currentValue, direction);
this.targetMode = 'grid';
this.echoBaseTick = null;
const geometryDirection = this.geometryDirectionForTarget(
this.targetTick,
direction,
);
if (!this.parkingApplied && !this.firstArmed) {
this.applyParking(geometryDirection);
}
if (this.firstArmed && !this.hasTicked) {
this.applyFirstTargetGeometry(this.targetTick, geometryDirection);
}
}
setDirectionAfterFirst(direction) {
this.movementDirection = direction;
this.targetMode = 'grid';
this.echoBaseTick = null;
this.targetTick = this.nextGridTick(this.currentValue, direction);
const geometryDirection = this.geometryDirectionForTarget(
this.targetTick,
direction,
);
this.applyNormalTargetGeometry(this.targetTick, geometryDirection);
}
handlePointerDown(event) {
const isTouch = event.pointerType === 'touch';
if (isTouch) {
if (event.target !== this.driver) return;
} else {
if (!this.surface.contains(event.target) || event.button !== 0) return;
event.preventDefault();
this.desktopPointer = true;
this.surface.classList.add('is-dragging');
try {
this.surface.setPointerCapture(event.pointerId);
} catch {
}
}
this.activePointerId = event.pointerId;
this.pointerMoves = 0;
this.previousPointerValue = this.valueFromPoint(event.clientX, event.clientY);
this.emitValue(this.previousPointerValue, false);
}
handlePointerMove(event) {
if (event.pointerId !== this.activePointerId) return;
this.pointerMoves += 1;
const nextValue = this.valueFromPoint(event.clientX, event.clientY);
this.emitValue(nextValue, false);
if (this.desktopPointer) {
const delta = nextValue - this.previousPointerValue;
if (Math.abs(delta) >= this.options.directionEpsilon) {
const direction = Math.sign(delta);
const tick = this.nextGridTick(this.previousPointerValue, direction);
if (this.crossed(this.previousPointerValue, nextValue, tick, direction)) {
const major = Boolean(this.options.isMajor(tick));
this.options.onTick({
value: tick,
kind: major ? 'major' : 'minor',
phase: 'desktop',
direction,
});
}
}
this.previousPointerValue = nextValue;
}
const now = performance.now();
if (now - this.lastDebugAt > 800) {
this.lastDebugAt = now;
this.debug(
`moves=${this.pointerMoves}/${this.touchMoves}; value=${Math.round(this.currentValue)}`,
);
}
}
finishPointer(event) {
if (event.pointerId !== this.activePointerId) return;
if (this.desktopPointer) {
this.value = this.currentValue;
this.surface.classList.remove('is-dragging');
try {
this.surface.releasePointerCapture(event.pointerId);
} catch {
}
}
this.desktopPointer = false;
this.activePointerId = null;
}
handleTouchStart(event) {
if (event.target !== this.driver || event.touches.length !== 1) return;
const touch = event.touches[0];
const rect = this.surface.getBoundingClientRect();
this.activeTouchId = touch.identifier;
this.gestureStartedAt = performance.now();
this.startClientX = touch.clientX;
this.startLocalFull = touch.clientX - rect.left;
this.previousTouchValue = this.valueFromPoint(
touch.clientX,
touch.clientY,
);
this.emitValue(this.previousTouchValue, false);
this.movementDirection = 0;
this.parkingApplied = false;
this.storedStartLocal = null;
this.firstArmed = false;
this.hasTicked = false;
this.visualStateOn = false;
this.targetTick = null;
this.targetMode = 'grid';
this.echoBaseTick = null;
this.pointerMoves = 0;
this.touchMoves = 0;
this.resetDriverFull();
clearTimeout(this.armTimer);
this.armTimer = setTimeout(
() => this.armFirstNow(),
this.options.firstArmMs,
);
this.debug(`start=${Math.round(this.previousTouchValue)}`);
}
handleTouchMove(event) {
if (this.activeTouchId === null) return;
const touch = touchById(event.touches, this.activeTouchId);
if (!touch) return;
this.touchMoves += 1;
const nextValue = this.valueFromPoint(touch.clientX, touch.clientY);
const delta = nextValue - this.previousTouchValue;
this.emitValue(nextValue, false);
if (Math.abs(delta) >= this.options.directionEpsilon) {
const nextDirection = Math.sign(delta);
if (nextDirection !== this.movementDirection) {
if (this.hasTicked) {
this.setDirectionAfterFirst(nextDirection);
} else {
this.setDirectionBeforeFirst(nextDirection);
}
}
}
if (
!this.firstArmed &&
this.movementDirection !== 0 &&
performance.now() - this.gestureStartedAt >= this.options.firstArmMs
) {
this.armFirstNow();
}
if (
this.firstArmed &&
this.movementDirection !== 0 &&
this.crossed(
this.previousTouchValue,
nextValue,
this.targetTick,
this.movementDirection,
)
) {
this.processTick(this.targetTick);
}
this.previousTouchValue = nextValue;
}
processTick(hitTick) {
if (hitTick === null) return;
this.emitValue(this.currentValue, true);
this.visualStateOn = !this.visualStateOn;
this.hasTicked = true;
if (this.targetMode === 'echo') {
const baseTick = this.echoBaseTick ?? hitTick;
this.options.onTick({
value: baseTick,
kind: 'major',
phase: 'echo',
direction: this.movementDirection,
});
const following = this.nextGridTick(
baseTick + this.movementDirection * 0.01,
this.movementDirection,
);
this.scheduleNormalTarget(
following,
this.movementDirection,
'grid',
null,
);
return;
}
const major = Boolean(this.options.isMajor(hitTick));
this.options.onTick({
value: hitTick,
kind: major ? 'major' : 'minor',
phase: 'primary',
direction: this.movementDirection,
});
const shouldAccent = major && this.options.accentOffset > 0;
if (shouldAccent) {
const echoTarget = clamp(
hitTick + this.movementDirection * this.options.accentOffset,
this.options.min,
this.options.max,
);
if (echoTarget !== hitTick) {
this.scheduleNormalTarget(
echoTarget,
this.movementDirection,
'echo',
hitTick,
);
return;
}
}
const following = this.nextGridTick(
hitTick + this.movementDirection * 0.01,
this.movementDirection,
);
this.scheduleNormalTarget(
following,
this.movementDirection,
'grid',
null,
);
}
finishTouch(event) {
if (this.activeTouchId === null) return;
if (touchById(event.touches, this.activeTouchId)) return;
this.debug(
`end=${Math.round(this.currentValue)}; ticked=${this.hasTicked}; checked=${this.driver.checked}`,
);
this.value = this.currentValue;
this.activeTouchId = null;
clearTimeout(this.armTimer);
clearTimeout(this.rearmTimer);
setTimeout(() => {
this.driver.checked = false;
this.resetDriverFull();
}, 140);
}
setValue(value) {
this.value = clamp(Number(value), this.options.min, this.options.max);
this.emitValue(this.value, false);
}
destroy() {
clearTimeout(this.armTimer);
clearTimeout(this.rearmTimer);
this.abortController.abort();
}
}
export function bindHapticTap(input, callback) {
if (!(input instanceof HTMLInputElement)) {
throw new TypeError('bindHapticTap expects an input element.');
}
input.type = 'checkbox';
input.setAttribute('switch', '');
const handler = (event) => {
callback?.({ event, checked: input.checked });
};
input.addEventListener('change', handler);
return () => input.removeEventListener('change', handler);
}
export function bindHapticToggle({ control, input, initial = false, onChange }) {
if (!(control instanceof HTMLElement)) {
throw new TypeError('bindHapticToggle expects a control element.');
}
if (!(input instanceof HTMLInputElement)) {
throw new TypeError('bindHapticToggle expects an input element.');
}
input.type = 'checkbox';
input.setAttribute('switch', '');
input.checked = Boolean(initial);
let checked = input.checked;
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
const commit = (next, event, source) => {
checked = Boolean(next);
input.checked = checked;
control.setAttribute('aria-checked', String(checked));
control.classList.toggle('is-on', checked);
onChange?.({ checked, event, source });
};
const nativeChange = (event) => commit(input.checked, event, 'native');
const desktopClick = (event) => {
if (!finePointer.matches || event.target === input) return;
commit(!checked, event, 'pointer');
};
const keydown = (event) => {
if (event.key !== ' ' && event.key !== 'Enter') return;
event.preventDefault();
commit(!checked, event, 'keyboard');
};
input.addEventListener('change', nativeChange);
control.addEventListener('click', desktopClick);
control.addEventListener('keydown', keydown);
commit(checked, new Event('init'), 'init');
return () => {
input.removeEventListener('change', nativeChange);
control.removeEventListener('click', desktopClick);
control.removeEventListener('keydown', keydown);
};
}
export function bindNativeToggle(input, callback) {
if (!(input instanceof HTMLInputElement)) {
throw new TypeError('bindNativeToggle expects an input element.');
}
input.type = 'checkbox';
input.setAttribute('switch', '');
const handler = (event) => {
callback?.({ event, checked: input.checked });
};
input.addEventListener('change', handler);
handler(new Event('change'));
return () => input.removeEventListener('change', handler);
}
