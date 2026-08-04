const XY_DEFAULTS = {
  columns: 10,
  rows: 8,
  x: 52,
  y: 46,
  firstArmMs: 228,
  parkWidth: 1200,
  parkHeight: 100,
  driverHeight: 76,
  onValue: () => {},
  onTick: () => {},
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

export class HapticXYPad {
  constructor(options) {
    this.options = { ...XY_DEFAULTS, ...options };
    this.surface = this.options.surface;
    this.driver = this.options.driver;

    if (!(this.surface instanceof HTMLElement)) {
      throw new TypeError('HapticXYPad requires a surface element.');
    }

    if (!(this.driver instanceof HTMLInputElement)) {
      throw new TypeError('HapticXYPad requires an input driver.');
    }

    this.driver.type = 'checkbox';
    this.driver.setAttribute('switch', '');

    this.x = clamp(Number(this.options.x), 0, 100);
    this.y = clamp(Number(this.options.y), 0, 100);
    this.cellX = this.cellFor(this.x, this.options.columns);
    this.cellY = this.cellFor(this.y, this.options.rows);

    this.activePointerId = null;
    this.activeTouchId = null;
    this.desktopPointer = false;
    this.nativeReady = false;
    this.hasTicked = false;
    this.visualStateOn = false;
    this.startClientX = 0;
    this.storedStartLocal = null;
    this.armTimer = null;
    this.parkTimer = null;

    this.abortController = new AbortController();
    this.signal = this.abortController.signal;
    this.finePointerQuery = matchMedia('(hover: hover) and (pointer: fine)');

    this.syncPointerMode();
    this.bind();
    this.resetDriver();
    this.emit(false);
  }

  bind() {
    window.addEventListener('pointerdown', event => this.pointerDown(event), {
      capture: true,
      signal: this.signal,
    });
    window.addEventListener('pointermove', event => this.pointerMove(event), {
      capture: true,
      signal: this.signal,
    });
    window.addEventListener('pointerup', event => this.pointerEnd(event), {
      capture: true,
      signal: this.signal,
    });
    window.addEventListener('pointercancel', event => this.pointerEnd(event), {
      capture: true,
      signal: this.signal,
    });

    this.surface.addEventListener('touchstart', event => this.touchStart(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });
    this.surface.addEventListener('touchmove', event => this.touchMove(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });
    this.surface.addEventListener('touchend', event => this.touchEnd(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });
    this.surface.addEventListener('touchcancel', event => this.touchEnd(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });

    this.finePointerQuery.addEventListener('change', () => this.syncPointerMode(), {
      signal: this.signal,
    });
  }

  syncPointerMode() {
    const fine = this.finePointerQuery.matches;
    this.surface.classList.toggle('is-fine-pointer', fine);
    this.driver.style.pointerEvents = fine ? 'none' : 'auto';
  }

  point(clientX, clientY) {
    const rect = this.surface.getBoundingClientRect();
    return {
      x: clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1) * 100,
      y: clamp((clientY - rect.top) / Math.max(1, rect.height), 0, 1) * 100,
      localX: clientX - rect.left,
    };
  }

  cellFor(value, count) {
    return clamp(Math.floor(value / 100 * count), 0, count - 1);
  }

  setPoint(point, animate = false) {
    this.x = point.x;
    this.y = point.y;
    this.options.onValue({
      x: this.x,
      y: this.y,
      ratioX: this.x / 100,
      ratioY: this.y / 100,
      animate,
    });
  }

  emit(animate) {
    this.options.onValue({
      x: this.x,
      y: this.y,
      ratioX: this.x / 100,
      ratioY: this.y / 100,
      animate,
    });
  }

  pointerDown(event) {
    if (event.pointerType === 'touch') return;
    if (event.button !== 0 || !this.surface.contains(event.target)) return;

    event.preventDefault();
    this.desktopPointer = true;
    this.activePointerId = event.pointerId;
    this.surface.classList.add('is-dragging');

    try {
      this.surface.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is an enhancement, not a requirement.
    }

    const point = this.point(event.clientX, event.clientY);
    this.setPoint(point, false);
    this.cellX = this.cellFor(point.x, this.options.columns);
    this.cellY = this.cellFor(point.y, this.options.rows);
  }

  pointerMove(event) {
    if (event.pointerId !== this.activePointerId || !this.desktopPointer) return;

    const point = this.point(event.clientX, event.clientY);
    const nextCellX = this.cellFor(point.x, this.options.columns);
    const nextCellY = this.cellFor(point.y, this.options.rows);
    const crossedGrid = nextCellX !== this.cellX || nextCellY !== this.cellY;

    this.setPoint(point, crossedGrid);

    if (crossedGrid) {
      this.options.onTick({
        x: point.x,
        y: point.y,
        cellX: nextCellX,
        cellY: nextCellY,
        source: 'desktop',
      });
    }

    this.cellX = nextCellX;
    this.cellY = nextCellY;
  }

  pointerEnd(event) {
    if (event.pointerId !== this.activePointerId) return;

    if (this.desktopPointer) {
      this.surface.classList.remove('is-dragging');
      try {
        this.surface.releasePointerCapture(event.pointerId);
      } catch {
        // Ignore unsupported release.
      }
    }

    this.desktopPointer = false;
    this.activePointerId = null;
  }

  resetDriver() {
    clearTimeout(this.parkTimer);
    Object.assign(this.driver.style, {
      left: '0px',
      right: 'auto',
      top: '0px',
      width: '100%',
      height: '100%',
      direction: 'ltr',
    });
    void this.driver.offsetWidth;
  }

  applyStartupParking(clientX) {
    const rect = this.surface.getBoundingClientRect();
    const localX = clientX - rect.left;
    const localRight = this.options.parkWidth - 1;

    this.storedStartLocal = localRight;

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${localX - localRight}px`,
      right: 'auto',
      top: '0px',
      width: `${this.options.parkWidth}px`,
      height: `${this.options.parkHeight}px`,
    });

    void this.driver.offsetWidth;
  }

  triggerFirstAt(clientX) {
    const rect = this.surface.getBoundingClientRect();
    const localX = clientX - rect.left;
    const width = this.options.parkWidth;
    const storedStart = this.storedStartLocal ?? width - 1;
    const changePosition = storedStart + 0.4 * width;
    const targetPosition = localX - 2;
    const requiredHeight = Math.max(20, width - storedStart + 12);

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${targetPosition - changePosition}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${requiredHeight}px`,
    });

    void this.driver.offsetWidth;
  }

  triggerNormalAt(clientX) {
    const rect = this.surface.getBoundingClientRect();
    const localX = clientX - rect.left;
    const width = 420;
    const midpoint = localX + (this.visualStateOn ? 2 : -2);

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${midpoint - width / 2}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${this.options.driverHeight}px`,
    });

    void this.driver.offsetWidth;
  }

  parkForState(clientX) {
    const rect = this.surface.getBoundingClientRect();
    const localX = clientX - rect.left;
    const width = this.options.parkWidth;
    const left = this.visualStateOn
      ? localX - (width - 1)
      : localX - 1;

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${left}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${this.options.parkHeight}px`,
    });

    void this.driver.offsetWidth;
  }

  touchStart(event) {
    if (event.target !== this.driver || event.touches.length !== 1) return;

    const touch = event.touches[0];
    const point = this.point(touch.clientX, touch.clientY);

    this.activeTouchId = touch.identifier;
    this.nativeReady = false;
    this.hasTicked = false;
    this.visualStateOn = false;
    this.cellX = this.cellFor(point.x, this.options.columns);
    this.cellY = this.cellFor(point.y, this.options.rows);
    this.setPoint(point, false);

    this.driver.checked = false;
    this.applyStartupParking(touch.clientX);

    clearTimeout(this.armTimer);
    this.armTimer = setTimeout(() => {
      if (this.activeTouchId !== null) this.nativeReady = true;
    }, this.options.firstArmMs);
  }

  touchMove(event) {
    if (this.activeTouchId === null) return;

    const touch = touchById(event.touches, this.activeTouchId);
    if (!touch) return;

    const point = this.point(touch.clientX, touch.clientY);
    const nextCellX = this.cellFor(point.x, this.options.columns);
    const nextCellY = this.cellFor(point.y, this.options.rows);
    const crossedGrid = nextCellX !== this.cellX || nextCellY !== this.cellY;

    this.setPoint(point, crossedGrid);

    if (crossedGrid && this.nativeReady) {
      if (this.hasTicked) {
        this.triggerNormalAt(touch.clientX);
      } else {
        this.triggerFirstAt(touch.clientX);
        this.hasTicked = true;
      }

      this.visualStateOn = !this.visualStateOn;

      this.options.onTick({
        x: point.x,
        y: point.y,
        cellX: nextCellX,
        cellY: nextCellY,
        source: 'touch',
      });

      clearTimeout(this.parkTimer);
      this.parkTimer = setTimeout(() => {
        if (this.activeTouchId !== null) this.parkForState(touch.clientX);
      }, 0);
    }

    this.cellX = nextCellX;
    this.cellY = nextCellY;
  }

  touchEnd(event) {
    if (this.activeTouchId === null) return;
    if (touchById(event.touches, this.activeTouchId)) return;

    this.activeTouchId = null;
    this.nativeReady = false;
    clearTimeout(this.armTimer);
    clearTimeout(this.parkTimer);

    setTimeout(() => {
      this.driver.checked = false;
      this.resetDriver();
    }, 120);
  }

  destroy() {
    this.abortController.abort();
    clearTimeout(this.armTimer);
    clearTimeout(this.parkTimer);
  }
}
