# Haptic Lab

Experimental browser-only haptic components for iPhone Safari.

## Structure

- `src/haptics.js` — reusable `HapticRange`, `bindHapticTap`, and `bindNativeToggle` APIs.
- `src/app.js` — demo wiring and animation state.
- `styles.css` — landing-page and component styling.
- `index.html` — static showcase, no build step required.

## Static usage

```html
<link rel="stylesheet" href="./styles.css">
<script type="module">
  import { HapticRange } from './src/haptics.js';

  new HapticRange({
    surface: document.querySelector('#surface'),
    driver: document.querySelector('#driver'),
    step: 10,
    onValue: ({ value }) => console.log(value),
    onTick: ({ value, kind }) => console.log(value, kind),
  });
</script>
```

The driver must remain a rendered native control:

```html
<input id="driver" type="checkbox" switch>
```

Do not apply `appearance: none` or `display: none` to the driver. The demo keeps it transparent with `opacity: 0` while preserving native WebKit rendering and direct trusted touch handling.

## Current scope

This is research-grade code for physical-device testing. The next packaging stage is a TypeScript core plus React wrappers and automated browser fallbacks.
