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

/**
 * Experimental iOS WebKit haptic range driver.
 *
 * The class keeps a real <input type="checkbox" switch> as the active touch
 * target. It uses WebKit's native switch pointer tracking rather than
 * synthetic clicks. The first threshold follows WebKit's special 0.4 × width
 * startup branch; later thresholds use the current renderer midpoint.
 */
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
    this.gestureStartedAt = 0;
    this.armTimer = null;
    this.rearmTimer = null;

    this.startClientX = 0;
    this.startLocalFull = 0;
    this.currentValue = this.value;
    this.previousTouchValue = this.value;
    this.movementDirection = 0;

    this.parkingApplied = false;
    this.storedStartLocal = null;
    this.capturedStartLocal = null;

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

    window.addEventListener(
      'resize',
      () => this.emitValue(this.currentValue, false),
      { signal: this.signal },
    );
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

  valueFromClientX(clientX) {
    const { rect, width } = this.metrics();
    const ratio = clamp(
      (clientX - rect.left - this.options.trackInset) / width,
      0,
      1,
    );

    return this.options.min + ratio * (this.options.max - this.options.min);
  }

  xFromValue(value) {
    const { left, width } = this.metrics();
    const ratio =
      (clamp(value, this.options.min, this.options.max) - this.options.min) /
      (this.options.max - this.options.min);

    return left + width * ratio;
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
      return next <= max ? next : null;
    }

    if (direction < 0) {
      const index = Math.ceil((value - min - 0.0001) / step) - 1;
      const next = min + index * step;
      return next >= min ? next : null;
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

  applyParking(direction) {
    const rect = this.surface.getBoundingClientRect();
    const startX = this.startClientX - rect.left;
    const parkLocalRight = this.options.parkWidth - 1;

    if (direction > 0) {
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
      `park ${direction > 0 ? 'right' : 'left'}; local=${this.storedStartLocal}`,
    );
  }

  applyFirstTargetGeometry(tick, direction) {
    if (tick === null || direction === 0) return;

    const targetX = this.xFromValue(tick);
    const storedStart = this.capturedStartLocal ?? this.startLocalFull;
    const height = this.options.driverHeight;

    let width;
    let changePosition;

    if (direction > 0) {
      width = Math.max(120, storedStart + 60);
      changePosition =
        storedStart + this.options.startOffsetProportion * width;

      Object.assign(this.driver.style, {
        direction: 'ltr',
        left: `${targetX - changePosition}px`,
        right: 'auto',
        width: `${width}px`,
        height: `${height}px`,
      });
    } else {
      width = Math.max(120, storedStart + height + 24);
      changePosition =
        storedStart - this.options.startOffsetProportion * width;

      Object.assign(this.driver.style, {
        direction: 'rtl',
        left: `${targetX - changePosition}px`,
        right: 'auto',
        width: `${width}px`,
        height: `${height}px`,
      });
    }

    void this.driver.offsetWidth;

    this.debug(
      `first target=${tick}; dir=${direction > 0 ? 'right' : 'left'}; width=${Math.round(width)}`,
    );
  }

  rtlForSafeApproach(direction, stateOn) {
    if (direction > 0) return stateOn;
    return !stateOn;
  }

  applyNormalTargetGeometry(tick, direction) {
    if (tick === null || direction === 0) return;

    const { left, right, width } = this.metrics();
    const ratio =
      (tick - this.options.min) /
      (this.options.max - this.options.min);
    const tickX = left + width * ratio;
    const useRTL = this.rtlForSafeApproach(direction, this.visualStateOn);

    this.driver.style.direction = useRTL ? 'rtl' : 'ltr';

    if (direction > 0) {
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
      `target=${tick}; dir=${direction > 0 ? 'right' : 'left'}; state=${this.visualStateOn ? 'on' : 'off'}; ${useRTL ? 'rtl' : 'ltr'}`,
    );
  }

  scheduleNormalTarget(tick, direction, mode = 'grid', echoBase = null) {
    clearTimeout(this.rearmTimer);

    this.rearmTimer = setTimeout(() => {
      this.targetTick = tick;
      this.targetMode = mode;
      this.echoBaseTick = echoBase;

      if (tick !== null) {
        this.applyNormalTargetGeometry(tick, direction);
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
    this.capturedStartLocal = this.parkingApplied
      ? this.storedStartLocal
      : this.startLocalFull;

    this.targetTick = this.nextGridTick(
      this.currentValue,
      this.movementDirection,
    );
    this.targetMode = 'grid';
    this.echoBaseTick = null;

    this.applyFirstTargetGeometry(this.targetTick, this.movementDirection);
    this.debug(`native ready; first=${this.targetTick}`);
  }

  setDirectionBeforeFirst(direction) {
    this.movementDirection = direction;

    const elapsed = performance.now() - this.gestureStartedAt;

    if (!this.firstArmed && elapsed < this.options.firstArmMs) {
      this.applyParking(direction);
    }

    if (!this.firstArmed && elapsed >= this.options.firstArmMs) {
      this.capturedStartLocal = this.parkingApplied
        ? this.storedStartLocal
        : this.startLocalFull;
      this.armFirstNow();
      return;
    }

    if (this.firstArmed && !this.hasTicked) {
      this.targetTick = this.nextGridTick(this.currentValue, direction);
      this.targetMode = 'grid';
      this.echoBaseTick = null;
      this.applyFirstTargetGeometry(this.targetTick, direction);
    }
  }

  setDirectionAfterFirst(direction) {
    this.movementDirection = direction;
    this.targetMode = 'grid';
    this.echoBaseTick = null;
    this.targetTick = this.nextGridTick(this.currentValue, direction);
    this.applyNormalTargetGeometry(this.targetTick, direction);
  }

  handlePointerDown(event) {
    if (event.pointerType !== 'touch' || event.target !== this.driver) return;

    this.activePointerId = event.pointerId;
    this.pointerMoves = 0;
    this.emitValue(this.valueFromClientX(event.clientX), false);
  }

  handlePointerMove(event) {
    if (event.pointerId !== this.activePointerId) return;

    this.pointerMoves += 1;
    this.emitValue(this.valueFromClientX(event.clientX), false);

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

    this.previousTouchValue = this.valueFromClientX(touch.clientX);
    this.emitValue(this.previousTouchValue, false);

    this.movementDirection = 0;
    this.parkingApplied = false;
    this.storedStartLocal = null;
    this.capturedStartLocal = null;

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

    const nextValue = this.valueFromClientX(touch.clientX);
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

/**
 * Makes a visible component haptic by placing a real native switch over it.
 * The returned cleanup function removes the listener.
 */
export function bindHapticTap(input, callback) {
  if (!(input instanceof HTMLInputElement)) {
    throw new TypeError('bindHapticTap expects an input element.');
  }

  input.type = 'checkbox';
  input.setAttribute('switch', '');

  const handler = (event) => {
    callback?.({
      event,
      checked: input.checked,
    });
  };

  input.addEventListener('change', handler);
  return () => input.removeEventListener('change', handler);
}

export function bindNativeToggle(input, callback) {
  if (!(input instanceof HTMLInputElement)) {
    throw new TypeError('bindNativeToggle expects an input element.');
  }

  input.type = 'checkbox';
  input.setAttribute('switch', '');

  const handler = (event) => {
    callback?.({
      event,
      checked: input.checked,
    });
  };

  input.addEventListener('change', handler);
  handler(new Event('change'));

  return () => input.removeEventListener('change', handler);
}
