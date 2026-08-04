import { HapticXYPad } from './xy-pad.js';

/*
 * WebKit's first switch-drag threshold is asymmetric: before the first toggle
 * it uses the touch-start position plus or minus 0.4 × the current switch
 * width. The original XY pad always parked the touch near the switch's right
 * edge, which made the useful rendered region depend on how far right the
 * gesture started.
 *
 * Mirror the parking geometry around the field centre. A gesture starting in
 * the left half extends the native switch to the right and uses the RTL start
 * branch; a gesture starting in the right half keeps the proven LTR branch.
 * After the first native tick, the existing midpoint-based re-arming remains
 * unchanged.
 */
HapticXYPad.prototype.applyStartupParking = function applyStartupParking(clientX) {
  const rect = this.surface.getBoundingClientRect();
  const localX = clientX - rect.left;
  const width = this.options.parkWidth;
  const startOnLeft = localX < rect.width / 2;

  this.startupEdge = startOnLeft ? 'left' : 'right';

  if (startOnLeft) {
    this.storedStartLocal = 1;
    Object.assign(this.driver.style, {
      direction: 'rtl',
      left: `${localX - this.storedStartLocal}px`,
      right: 'auto',
    });
  } else {
    this.storedStartLocal = width - 1;
    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${localX - this.storedStartLocal}px`,
      right: 'auto',
    });
  }

  Object.assign(this.driver.style, {
    top: '0px',
    width: `${width}px`,
    height: `${Math.max(this.options.parkHeight, this.surface.clientHeight)}px`,
  });

  void this.driver.offsetWidth;
};

HapticXYPad.prototype.triggerFirstAt = function triggerFirstAt(clientX) {
  const rect = this.surface.getBoundingClientRect();
  const localX = clientX - rect.left;
  const width = this.options.parkWidth;
  const storedStart = this.storedStartLocal ?? width - 1;
  const startOnLeft = this.startupEdge === 'left';
  const surfaceHeight = Math.max(20, this.surface.clientHeight);

  if (startOnLeft) {
    // checked=false + RTL means the logical thumb starts on the right.
    // Move the current finger two pixels to the left of WebKit's special
    // start threshold: storedStart - 0.4 × width.
    const changePosition = storedStart - 0.4 * width;
    const targetLocal = changePosition - 2;
    const maximumSpecialBranchHeight = Math.max(20, width - storedStart - 12);

    Object.assign(this.driver.style, {
      direction: 'rtl',
      left: `${localX - targetLocal}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${Math.min(surfaceHeight, maximumSpecialBranchHeight)}px`,
    });
  } else {
    // Proven right-half branch: logical thumb starts on the left, so move the
    // finger two pixels to the right of storedStart + 0.4 × width.
    const changePosition = storedStart + 0.4 * width;
    const targetLocal = changePosition + 2;
    const requiredHeight = Math.max(
      surfaceHeight,
      width - storedStart + 12,
    );

    Object.assign(this.driver.style, {
      direction: 'ltr',
      left: `${localX - targetLocal}px`,
      right: 'auto',
      top: '0px',
      width: `${width}px`,
      height: `${requiredHeight}px`,
    });
  }

  void this.driver.offsetWidth;
};
