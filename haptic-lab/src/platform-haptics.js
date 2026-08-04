import { HapticRange } from './haptics.js';

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function touchById(touchList, identifier) {
  for (const touch of touchList) {
    if (touch.identifier === identifier) return touch;
  }
  return null;
}

function emitDesktopTicks(instance, fromValue, toValue) {
  const direction = Math.sign(toValue - fromValue);
  if (!direction) return;

  instance.movementDirection = direction;
  let probe = fromValue;
  let guard = 0;

  while (guard < 256) {
    guard += 1;
    const tick = instance.nextGridTick(probe, direction);
    if (tick === null || !instance.crossed(probe, toValue, tick, direction)) break;

    const major = Boolean(instance.options.isMajor(tick));
    instance.options.onTick({
      value: tick,
      kind: major ? 'major' : 'minor',
      phase: 'primary',
      direction,
      native: false,
    });
    instance.emitValue(toValue, true);

    if (major && instance.options.accentOffset > 0) {
      window.setTimeout(() => {
        instance.options.onTick({
          value: tick,
          kind: 'major',
          phase: 'echo',
          direction,
          native: false,
        });
      }, 76);
    }

    probe = tick + direction * 0.0001;
  }
}

/**
 * Keeps the proven iPhone HapticRange implementation intact, while adding a
 * normal mouse/trackpad path for desktop browsers.
 */
export class CrossPlatformRange extends HapticRange {
  constructor(options) {
    super(options);
    this.activePointerType = null;
    this.previousDesktopValue = this.currentValue;

    this.driver.addEventListener(
      'click',
      (event) => {
        if (navigator.maxTouchPoints === 0) {
          event.preventDefault();
          this.driver.checked = false;
        }
      },
      { signal: this.signal },
    );
  }

  handlePointerDown(event) {
    if (event.target !== this.driver) return;

    if (event.pointerType === 'touch') {
      super.handlePointerDown(event);
      this.activePointerType = 'touch';
      this.surface.classList.add('is-dragging');
      return;
    }

    event.preventDefault();
    this.activePointerId = event.pointerId;
    this.activePointerType = event.pointerType;
    this.pointerMoves = 0;
    this.previousDesktopValue = this.valueFromClientX(event.clientX);
    this.emitValue(this.previousDesktopValue, false);
    this.surface.classList.add('is-dragging');

    try {
      this.driver.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is only an enhancement; window listeners still work.
    }
  }

  handlePointerMove(event) {
    if (event.pointerId !== this.activePointerId) return;

    if (this.activePointerType === 'touch') {
      super.handlePointerMove(event);
      return;
    }

    event.preventDefault();
    this.pointerMoves += 1;
    const nextValue = this.valueFromClientX(event.clientX);
    emitDesktopTicks(this, this.previousDesktopValue, nextValue);
    this.previousDesktopValue = nextValue;
    this.emitValue(nextValue, false);
  }

  finishPointer(event) {
    if (event.pointerId !== this.activePointerId) return;

    if (this.activePointerType === 'touch') {
      super.finishPointer(event);
      this.activePointerType = null;
      this.surface.classList.remove('is-dragging');
      return;
    }

    event.preventDefault();
    try {
      this.driver.releasePointerCapture(event.pointerId);
    } catch {
      // Browsers may release capture automatically before pointerup.
    }

    this.value = this.currentValue;
    this.driver.checked = false;
    this.activePointerId = null;
    this.activePointerType = null;
    this.surface.classList.remove('is-dragging');
  }

  finishTouch(event) {
    super.finishTouch(event);
    if (!touchById(event.touches, this.activeTouchId)) {
      this.surface.classList.remove('is-dragging');
    }
  }
}

/**
 * True angular dial. Desktop follows the pointer angle around the centre.
 * Touch keeps the native switch driver and projects every angular detent onto
 * the switch's horizontal tracking axis.
 */
export class AngularHapticDial extends CrossPlatformRange {
  constructor(options) {
    super({
      arcStart: -135,
      arcEnd: 135,
      geometryWidth: 420,
      geometryHeight: 58,
      minimumRadiusRatio: 0.24,
      ...options,
      trackInset: 0,
    });

    this.arcStart = Number(options.arcStart ?? -135);
    this.arcEnd = Number(options.arcEnd ?? 135);
    this.geometryWidth = Number(options.geometryWidth ?? 420);
    this.geometryHeight = Number(options.geometryHeight ?? 58);
    this.minimumRadiusRatio = Number(options.minimumRadiusRatio ?? 0.24);
    this.touchRadius = 0;
    this.currentTouchLocalX = 0;
    this.switchDirection = 0;
  }

  dialMetrics() {
    const rect = this.surface.getBoundingClientRect();
    return {
      rect,
      centerX: rect.width / 2,
      centerY: rect.height / 2,
      radius: Math.min(rect.width, rect.height) / 2,
    };
  }

  valueToAngle(value) {
    const ratio =
      (clamp(value, this.options.min, this.options.max) - this.options.min) /
      (this.options.max - this.options.min);
    return this.arcStart + ratio * (this.arcEnd - this.arcStart);
  }

  pointFromClient(clientX, clientY) {
    const { rect, centerX, centerY, radius } = this.dialMetrics();
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    const dx = localX - centerX;
    const dy = localY - centerY;
    const pointRadius = Math.hypot(dx, dy);

    if (pointRadius < radius * this.minimumRadiusRatio) {
      return {
        value: this.currentValue,
        angle: this.valueToAngle(this.currentValue),
        radius: Math.max(pointRadius, radius * 0.68),
        localX,
      };
    }

    let angle = (Math.atan2(dx, -dy) * 180) / Math.PI;
    if (angle < this.arcStart || angle > this.arcEnd) {
      angle = dx < 0 ? this.arcStart : this.arcEnd;
    }
    angle = clamp(angle, this.arcStart, this.arcEnd);

    const ratio = (angle - this.arcStart) / (this.arcEnd - this.arcStart);
    return {
      value: this.options.min + ratio * (this.options.max - this.options.min),
      angle,
      radius: pointRadius,
      localX,
    };
  }

  targetLocalX(value) {
    const { centerX, radius } = this.dialMetrics();
    const safeRadius = clamp(
      this.touchRadius || radius * 0.68,
      radius * 0.34,
      radius * 0.92,
    );
    const angle = (this.valueToAngle(value) * Math.PI) / 180;
    return centerX + safeRadius * Math.sin(angle);
  }

  xFromValue(value) {
    return this.targetLocalX(value);
  }

  switchDirectionForTarget(tick, angularDirection) {
    const sampleAngle = this.valueToAngle(tick) - angularDirection * 0.8;
    const tangent = Math.cos((sampleAngle * Math.PI) / 180) * angularDirection;
    const fallback = Math.sign(this.targetLocalX(tick) - this.currentTouchLocalX);
    return Math.sign(tangent) || fallback || angularDirection;
  }

  resetDriverFull() {
    clearTimeout(this.rearmTimer);
    Object.assign(this.driver.style, {
      left: '0px',
      right: 'auto',
      top: '0px',
      width: '100%',
      height: `${Math.max(1, this.surface.clientHeight)}px`,
      direction: 'ltr',
      writingMode: 'horizontal-tb',
    });
    void this.driver.offsetWidth;
  }

  applyNormalTargetGeometry(tick, angularDirection) {
    if (tick === null || angularDirection === 0) return;

    const targetX = this.targetLocalX(tick);
    const switchDirection = this.switchDirectionForTarget(tick, angularDirection);
    const useRTL = this.rtlForSafeApproach(switchDirection, this.visualStateOn);
    const width = Math.max(this.geometryWidth, this.surface.clientWidth * 1.18);

    this.switchDirection = switchDirection;
    Object.assign(this.driver.style, {
      direction: useRTL ? 'rtl' : 'ltr',
      left: `${targetX - width / 2}px`,
      right: 'auto',
      top: `${(this.surface.clientHeight - this.geometryHeight) / 2}px`,
      width: `${width}px`,
      height: `${this.geometryHeight}px`,
    });
    void this.driver.offsetWidth;
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
    this.targetTick = this.nextGridTick(this.currentValue, this.movementDirection);
    this.targetMode = 'grid';
    this.echoBaseTick = null;
    if (this.targetTick === null) return;

    this.switchDirection = this.switchDirectionForTarget(
      this.targetTick,
      this.movementDirection,
    );
    super.applyFirstTargetGeometry(this.targetTick, this.switchDirection);
  }

  setDirectionBeforeFirst(direction) {
    this.movementDirection = direction;
    const target = this.nextGridTick(this.currentValue, direction);
    if (target === null) return;

    this.switchDirection = this.switchDirectionForTarget(target, direction);
    if (!this.parkingApplied && !this.firstArmed) {
      this.applyParking(this.switchDirection);
    }

    if (this.firstArmed && !this.hasTicked) {
      this.targetTick = target;
      this.targetMode = 'grid';
      this.echoBaseTick = null;
      super.applyFirstTargetGeometry(this.targetTick, this.switchDirection);
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
    if (event.target !== this.driver) return;

    const point = this.pointFromClient(event.clientX, event.clientY);
    this.activePointerId = event.pointerId;
    this.activePointerType = event.pointerType;
    this.previousDesktopValue = point.value;
    this.emitValue(point.value, false);
    this.surface.classList.add('is-dragging');

    if (event.pointerType !== 'touch') {
      event.preventDefault();
      try {
        this.driver.setPointerCapture(event.pointerId);
      } catch {
        // Optional enhancement.
      }
    }
  }

  handlePointerMove(event) {
    if (event.pointerId !== this.activePointerId) return;

    const point = this.pointFromClient(event.clientX, event.clientY);
    if (this.activePointerType !== 'touch') {
      event.preventDefault();
      emitDesktopTicks(this, this.previousDesktopValue, point.value);
      this.previousDesktopValue = point.value;
    }
    this.emitValue(point.value, false);
  }

  handleTouchStart(event) {
    if (event.target !== this.driver || event.touches.length !== 1) return;

    const touch = event.touches[0];
    const rect = this.surface.getBoundingClientRect();
    const point = this.pointFromClient(touch.clientX, touch.clientY);

    this.activeTouchId = touch.identifier;
    this.gestureStartedAt = performance.now();
    this.startClientX = touch.clientX;
    this.startLocalFull = touch.clientX - rect.left;
    this.touchRadius = point.radius;
    this.currentTouchLocalX = point.localX;
    this.previousTouchValue = point.value;
    this.emitValue(point.value, false);

    this.movementDirection = 0;
    this.switchDirection = 0;
    this.parkingApplied = false;
    this.storedStartLocal = null;
    this.firstArmed = false;
    this.hasTicked = false;
    this.visualStateOn = false;
    this.targetTick = null;
    this.targetMode = 'grid';
    this.echoBaseTick = null;

    this.resetDriverFull();
    clearTimeout(this.armTimer);
    this.armTimer = setTimeout(() => this.armFirstNow(), this.options.firstArmMs);
  }

  handleTouchMove(event) {
    if (this.activeTouchId === null) return;

    const touch = touchById(event.touches, this.activeTouchId);
    if (!touch) return;

    const point = this.pointFromClient(touch.clientX, touch.clientY);
    const nextValue = point.value;
    const delta = nextValue - this.previousTouchValue;

    this.touchRadius = this.touchRadius
      ? this.touchRadius * 0.82 + point.radius * 0.18
      : point.radius;
    this.currentTouchLocalX = point.localX;
    this.emitValue(nextValue, false);

    if (Math.abs(delta) >= this.options.directionEpsilon) {
      const nextDirection = Math.sign(delta);
      if (nextDirection !== this.movementDirection) {
        if (this.hasTicked) this.setDirectionAfterFirst(nextDirection);
        else this.setDirectionBeforeFirst(nextDirection);
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
}
