const DEFAULTS = {
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

/**
 * Two-dimensional control surface with an experimental native iOS haptic grid.
 *
 * Desktop uses ordinary Pointer Events. On iPhone a transparent, rendered
 * <input type="checkbox" switch> owns the physical gesture. Every grid-line
 * crossing moves the switch threshold through the current finger position,
 * allowing WebKit to produce one trusted system tick without synthetic clicks.
 */
export class HapticXYPad {
  constructor(options) {
    this.options = { ...DEFAULTS, ...options };
    this.surface = this.options.surface;
    this.driver = this.options.driver;

    if (!(this.surface instanceof HTMLElement)) {
      throw new TypeError('HapticXYPad requires an HTMLElement surface.');
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
    window.addEventListener('pointerdown', (event) => this.pointerDown(event), {
      capture: true,
      signal: this.signal,
    });

    window.addEventListener('pointermove', (event) => this.pointerMove(event), {
      capture: true,
      signal: this.signal,
    });

    window.addEventListener('pointerup', (event) => this.pointerEnd(event), {
      capture: true,
      signal: this.signal,
    });

    window.addEventListener('pointercancel', (event) => this.pointerEnd(event), {
      capture: true,
      signal: this.signal,
    });

    this.surface.addEventListener('touchstart', (event) => this.touchStart(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });

    this.surface.addEventListener('touchmove', (event) => this.touchMove(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });

    this.surface.addEventListener('touchend', (event) => this.touchEnd(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });

    this.surface.addEventListener('touchcancel', (event) => this.touchEnd(event), {
      capture: true,
      passive: true,
      signal: this.signal,
    });

    this.surface.addEventListener('keydown', (event) => this.keyDown(event), {
      signal: this.signal,
    });

    this.finePointerQuery.addEventListener('change', () => this.syncPointerMode(), {
      signal: this.signal,
    });
  }

  syncPointerMode() {
    const desktopOnly = this.finePointerQuery.matches && navigator.maxTouchPoints === 0;
    this.surface.classList.toggle('is-fine-pointer', desktopOnly);
    this.driver.style.pointerEvents = desktopOnly ? 'none' : 'auto';
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
    return clamp(Math.floor((value / 100) * count), 0, count - 1);
  }

  setPoint(point, animate = false) {
    this.x = clamp(point.x, 0, 100);
    this.y = clamp(point.y, 0, 100);
    this.emit(animate);
  }

  setValue(x, y, animate = false) {
    const nextCellX = this.cellFor(x, this.options.columns);
    const nextCellY = this.cellFor(y, this.options.rows);
    const crossedX = nextCellX !== this.cellX;
    const crossedY = nextCellY !== this.cellY;

    this.setPoint({ x, y }, animate || crossedX || crossedY);

    if (crossedX || crossedY) {
      this.options.onTick({
        x: this.x,
        y: this.y,
        cellX: nextCellX,
        cellY: nextCellY,
        crossedX,
        crossedY,
        axis: crossedX && crossedY ? 'xy' : crossedX ? 'x' : 'y',
        source: 'keyboard',
        native: false,
      });
    }

    this.cellX = nextCellX;
    this.cellY = nextCellY;
  }

  emit(animate) {
    this.options.onValue({
      x: this.x,
      y: this.y,
      ratioX: this.x / 100,
      ratioY: this.y / 100,
      cellX: this.cellFor(this.x, this.options.columns),
      cellY: this.cellFor(this.y, this.options.rows),
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
      // Pointer capture is a progressive enhancement.
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
    const crossedX = nextCellX !== this.cellX;
    const crossedY = nextCellY !== this.cellY;
    const crossedGrid = crossedX || crossedY;

    this.setPoint(point, crossedGrid);

    if (crossedGrid) {
      this.options.onTick({
        x: point.x,
        y: point.y,
        cellX: nextCellX,
        cellY: nextCellY,
        crossedX,
        crossedY,
        axis: crossedX && crossedY ? 'xy' : crossedX ? 'x' : 'y',
        source: 'desktop',
        native: false,
      });
    }

    this.cellX = nextCellX;
    this.cellY = nextCellY;
  }

  pointerEnd(event) {
    if (event.pointerId !== this.activePointerId) return;

    this.surface.classList.remove('is-dragging');

    try {
      this.surface.releasePointerCapture(event.pointerId);
    } catch {
      // Ignore unsupported release.
    }

    this.desktopPointer = false;
    this.activePointerId = null;
  }

  keyDown(event) {
    const stepX = 100 / this.options.columns;
    const stepY = 100 / this.options.rows;
    let nextX = this.x;
    let nextY = this.y;

    if (event.key === 'ArrowLeft') nextX -= stepX;
    else if (event.key === 'ArrowRight') nextX += stepX;
    else if (event.key === 'ArrowUp') nextY -= stepY;
    else if (event.key === 'ArrowDown') nextY += stepY;
    else return;

    event.preventDefault();
    this.setValue(clamp(nextX, 0, 100), clamp(nextY, 0, 100), true);
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
      height: `${Math.max(this.options.parkHeight, this.surface.clientHeight)}px`,
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
    const requiredHeight = Math.max(
      this.surface.clientHeight,
      20,
      width - storedStart + 12,
    );

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
    const width = Math.max(420, this.surface.clientWidth + 96);
    const midpoint = localX + (this.visualStateOn ? 2 : -2);

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${midpoint - width / 2}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${Math.max(this.options.driverHeight, this.surface.clientHeight)}px`,
    });

    void this.driver.offsetWidth;
  }

  parkForState(clientX) {
    const rect = this.surface.getBoundingClientRect();
    const localX = clientX - rect.left;
    const width = this.options.parkWidth;
    const left = this.visualStateOn ? localX - (width - 1) : localX - 1;

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${left}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${Math.max(this.options.parkHeight, this.surface.clientHeight)}px`,
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
    const crossedX = nextCellX !== this.cellX;
    const crossedY = nextCellY !== this.cellY;
    const crossedGrid = crossedX || crossedY;

    this.setPoint(point, crossedGrid);

    if (crossedGrid && this.nativeReady) {
      if (this.hasTicked) this.triggerNormalAt(touch.clientX);
      else {
        this.triggerFirstAt(touch.clientX);
        this.hasTicked = true;
      }

      this.visualStateOn = !this.visualStateOn;

      this.options.onTick({
        x: point.x,
        y: point.y,
        cellX: nextCellX,
        cellY: nextCellY,
        crossedX,
        crossedY,
        axis: crossedX && crossedY ? 'xy' : crossedX ? 'x' : 'y',
        source: 'touch',
        native: true,
      });

      const clientX = touch.clientX;
      clearTimeout(this.parkTimer);
      this.parkTimer = setTimeout(() => {
        if (this.activeTouchId !== null) this.parkForState(clientX);
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
