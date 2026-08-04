(() => {
  "use strict";

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function isProbablyIOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent)
      || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  }

  function androidFallback(pattern = 8) {
    if (isProbablyIOS()) return;
    if (typeof navigator.vibrate !== "function") return;

    try {
      navigator.vibrate(pattern);
    } catch {
      // Progressive enhancement: visual interaction still works.
    }
  }

  class Toast {
    constructor(element) {
      this.element = element;
      this.title = element?.querySelector("[data-toast-title]");
      this.copy = element?.querySelector("[data-toast-copy]");
      this.timer = null;
    }

    show(title, copy) {
      if (!this.element) return;

      this.title.textContent = title;
      this.copy.textContent = copy;
      this.element.classList.add("is-visible");

      clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        this.element.classList.remove("is-visible");
      }, 1800);
    }
  }

  class HapticRange {
    constructor(root, options = {}) {
      if (!(root instanceof HTMLElement)) {
        throw new TypeError("HapticRange root must be an HTMLElement.");
      }

      this.root = root;
      this.track = root.querySelector("[data-range-track]");
      this.driver = root.querySelector("[data-range-driver]");
      this.fill = root.querySelector("[data-range-fill]");
      this.aura = root.querySelector("[data-range-aura]");
      this.thumb = root.querySelector("[data-range-thumb]");
      this.output = root.querySelector("[data-range-value]");
      this.ticksLayer = root.querySelector("[data-range-ticks]");
      this.status = root.querySelector("[data-range-status]");

      if (!this.track || !this.driver || !this.fill || !this.thumb || !this.output) {
        throw new Error("HapticRange markup is incomplete.");
      }

      this.step = Number(options.step ?? root.dataset.step ?? 10);
      this.majorStep = Number(options.majorStep ?? root.dataset.majorStep ?? 20);
      this.pattern = String(options.pattern ?? root.dataset.pattern ?? "single");
      this.driverHeight = Number(options.driverHeight ?? root.dataset.driverHeight ?? 72);
      this.accentGap = Number(options.accentGap ?? root.dataset.accentGap ?? 1.6);
      this.value = Number(options.initial ?? root.dataset.initial ?? 50);

      this.trackInset = 18;
      this.parkWidth = 1200;
      this.parkHeight = 100;
      this.parkLocalLeft = 1;
      this.parkLocalRight = this.parkWidth - 1;
      this.startOffsetProportion = 0.4;
      this.firstArmMs = 228;
      this.directionEpsilon = 0.55;

      this.activePointerId = null;
      this.activeTouchId = null;
      this.gestureStartedAt = 0;
      this.armTimer = null;
      this.rearmTimer = null;

      this.startClientX = 0;
      this.startLocalFull = 0;
      this.previousTouchValue = this.value;
      this.movementDirection = 0;

      this.parkingApplied = false;
      this.storedStartLocal = null;
      this.firstArmed = false;
      this.hasTicked = false;
      this.visualStateOn = false;
      this.target = null;

      this.latestClientX = null;
      this.pointerMoves = 0;
      this.touchMoves = 0;

      this.bound = {
        pointerDown: this.onPointerDown.bind(this),
        pointerMove: this.onPointerMove.bind(this),
        pointerEnd: this.onPointerEnd.bind(this),
        touchStart: this.onTouchStart.bind(this),
        touchMove: this.onTouchMove.bind(this),
        touchEnd: this.onTouchEnd.bind(this),
        resize: this.onResize.bind(this),
      };

      this.buildTicks();
      this.bind();
      this.resetDriverFull();
      this.render(this.value);
    }

    buildTicks() {
      if (!this.ticksLayer) return;

      this.ticksLayer.textContent = "";

      for (let value = 0; value <= 100.0001; value += this.step) {
        const rounded = Number(value.toFixed(4));
        const tick = document.createElement("span");
        const isMajor = this.isMajor(rounded);

        tick.className = `range-tick${isMajor ? " is-major" : ""}`;
        tick.style.left = `${rounded}%`;
        tick.dataset.value = String(rounded);
        this.ticksLayer.appendChild(tick);
      }
    }

    bind() {
      window.addEventListener("pointerdown", this.bound.pointerDown, true);
      window.addEventListener("pointermove", this.bound.pointerMove, true);
      window.addEventListener("pointerup", this.bound.pointerEnd, true);
      window.addEventListener("pointercancel", this.bound.pointerEnd, true);
      window.addEventListener("resize", this.bound.resize);

      this.track.addEventListener("touchstart", this.bound.touchStart, {
        capture: true,
        passive: true,
      });

      this.track.addEventListener("touchmove", this.bound.touchMove, {
        capture: true,
        passive: true,
      });

      this.track.addEventListener("touchend", this.bound.touchEnd, {
        capture: true,
        passive: true,
      });

      this.track.addEventListener("touchcancel", this.bound.touchEnd, {
        capture: true,
        passive: true,
      });

      this.driver.addEventListener("change", (event) => {
        if (event.isTrusted) androidFallback(8);
      });
    }

    destroy() {
      window.removeEventListener("pointerdown", this.bound.pointerDown, true);
      window.removeEventListener("pointermove", this.bound.pointerMove, true);
      window.removeEventListener("pointerup", this.bound.pointerEnd, true);
      window.removeEventListener("pointercancel", this.bound.pointerEnd, true);
      window.removeEventListener("resize", this.bound.resize);

      this.track.removeEventListener("touchstart", this.bound.touchStart, true);
      this.track.removeEventListener("touchmove", this.bound.touchMove, true);
      this.track.removeEventListener("touchend", this.bound.touchEnd, true);
      this.track.removeEventListener("touchcancel", this.bound.touchEnd, true);
    }

    metrics() {
      const rect = this.track.getBoundingClientRect();
      const width = Math.max(1, rect.width - this.trackInset * 2);

      return {
        rect,
        left: this.trackInset,
        right: rect.width - this.trackInset,
        width,
      };
    }

    valueFromClientX(clientX) {
      const { rect, width } = this.metrics();

      return clamp(
        ((clientX - rect.left - this.trackInset) / width) * 100,
        0,
        100,
      );
    }

    xFromValue(value) {
      const { left, width } = this.metrics();
      return left + width * clamp(value, 0, 100) / 100;
    }

    render(value, intensity = null) {
      this.value = clamp(Number(value), 0, 100);

      const { width } = this.metrics();
      const x = this.trackInset + width * this.value / 100;
      const fillWidth = width * this.value / 100;

      this.fill.style.width = `${fillWidth}px`;
      this.thumb.style.left = `${x}px`;
      this.output.value = String(Math.round(this.value));
      this.output.textContent = String(Math.round(this.value));

      if (this.aura) {
        this.aura.style.left = `${Math.max(0, fillWidth - 40)}px`;
      }

      if (intensity) {
        const className = intensity === "major" ? "is-major-hit" : "is-hit";
        this.thumb.classList.remove("is-hit", "is-major-hit");
        void this.thumb.offsetWidth;
        this.thumb.classList.add(className);
      }
    }

    setStatus(text, armed = false) {
      if (this.status) {
        const dot = this.status.querySelector("i");
        this.status.replaceChildren(dot ?? document.createElement("i"), document.createTextNode(` ${text}`));
      }

      this.root.classList.toggle("is-armed", armed);
    }

    isMajor(value) {
      if (!Number.isFinite(this.majorStep) || this.majorStep <= 0) return false;

      const mod = Math.abs(value % this.majorStep);
      return mod < 0.001 || Math.abs(mod - this.majorStep) < 0.001;
    }

    touchById(touchList, identifier) {
      for (const touch of touchList) {
        if (touch.identifier === identifier) return touch;
      }
      return null;
    }

    resetDriverFull() {
      clearTimeout(this.rearmTimer);

      this.driver.style.left = "0px";
      this.driver.style.right = "auto";
      this.driver.style.width = "100%";
      this.driver.style.height = `${this.driverHeight}px`;
      this.driver.style.direction = "ltr";

      void this.driver.offsetWidth;
    }

    nextGridValue(value, direction) {
      const epsilon = 0.0001;

      if (direction > 0) {
        const next = (Math.floor((value + epsilon) / this.step) + 1) * this.step;
        return next <= 100 - this.step + epsilon
          ? Number(next.toFixed(4))
          : null;
      }

      if (direction < 0) {
        const next = (Math.ceil((value - epsilon) / this.step) - 1) * this.step;
        return next >= this.step - epsilon
          ? Number(next.toFixed(4))
          : null;
      }

      return null;
    }

    makeGridTarget(value) {
      if (value === null) return null;

      return {
        value,
        kind: "grid",
        gridValue: value,
      };
    }

    makeAccentTarget(gridValue, direction) {
      const value = gridValue + direction * this.accentGap;

      if (value <= this.step * 0.4 || value >= 100 - this.step * 0.4) {
        return null;
      }

      return {
        value,
        kind: "accent",
        gridValue,
      };
    }

    applyParking(direction) {
      const rect = this.track.getBoundingClientRect();
      const startX = this.startClientX - rect.left;

      if (direction > 0) {
        this.driver.style.direction = "ltr";
        this.driver.style.left = `${startX - this.parkLocalRight}px`;
        this.driver.style.right = "auto";
        this.storedStartLocal = this.parkLocalRight;
      } else {
        this.driver.style.direction = "rtl";
        this.driver.style.left = `${startX - this.parkLocalLeft}px`;
        this.driver.style.right = "auto";
        this.storedStartLocal = this.parkLocalLeft;
      }

      this.driver.style.width = `${this.parkWidth}px`;
      this.driver.style.height = `${this.parkHeight}px`;

      this.parkingApplied = true;
      void this.driver.offsetWidth;
    }

    effectiveStoredStartLocal() {
      return this.parkingApplied
        ? this.storedStartLocal
        : this.startLocalFull;
    }

    applyFirstTargetGeometry(target, direction) {
      if (!target || direction === 0) return;

      const targetX = this.xFromValue(target.value);
      const storedStart = this.effectiveStoredStartLocal();
      const width = this.parkWidth;

      if (direction > 0) {
        const requiredHeight = Math.max(
          20,
          width - storedStart + 12,
        );

        const changePosition =
          storedStart + this.startOffsetProportion * width;

        this.driver.style.direction = "ltr";
        this.driver.style.left = `${targetX - changePosition}px`;
        this.driver.style.right = "auto";
        this.driver.style.width = `${width}px`;
        this.driver.style.height = `${requiredHeight}px`;
      } else {
        const maxHeight = Math.max(
          20,
          Math.min(120, (width - storedStart) / 2),
        );

        const changePosition =
          storedStart - this.startOffsetProportion * width;

        this.driver.style.direction = "rtl";
        this.driver.style.left = `${targetX - changePosition}px`;
        this.driver.style.right = "auto";
        this.driver.style.width = `${width}px`;
        this.driver.style.height = `${maxHeight}px`;
      }

      void this.driver.offsetWidth;
    }

    rtlForSafeApproach(direction, stateOn) {
      return direction > 0 ? stateOn : !stateOn;
    }

    applyNormalTargetGeometry(target, direction) {
      if (!target || direction === 0) return;

      const { left, right, width } = this.metrics();
      const targetX = left + width * target.value / 100;
      const useRTL = this.rtlForSafeApproach(direction, this.visualStateOn);

      this.driver.style.direction = useRTL ? "rtl" : "ltr";

      if (direction > 0) {
        this.driver.style.left = `${left}px`;
        this.driver.style.right = "auto";
        this.driver.style.width = `${
          Math.max(this.driverHeight + 2, 2 * (targetX - left))
        }px`;
      } else {
        this.driver.style.left = "auto";
        this.driver.style.right = `${this.track.clientWidth - right}px`;
        this.driver.style.width = `${
          Math.max(this.driverHeight + 2, 2 * (right - targetX))
        }px`;
      }

      this.driver.style.height = `${this.driverHeight}px`;
      void this.driver.offsetWidth;
    }

    scheduleNormalTarget(target, direction) {
      clearTimeout(this.rearmTimer);

      this.rearmTimer = setTimeout(() => {
        this.target = target;
        this.applyNormalTargetGeometry(this.target, direction);
      }, 0);
    }

    crossed(fromValue, toValue, target, direction) {
      if (!target) return false;
      if (direction > 0) return fromValue < target.value && toValue >= target.value;
      if (direction < 0) return fromValue > target.value && toValue <= target.value;
      return false;
    }

    armFirstNow() {
      if (
        this.activeTouchId === null
        || this.movementDirection === 0
        || this.firstArmed
      ) {
        return;
      }

      this.firstArmed = true;
      this.target = this.makeGridTarget(
        this.nextGridValue(this.value, this.movementDirection),
      );

      this.applyFirstTargetGeometry(this.target, this.movementDirection);
      this.setStatus("Engaged · move freely", true);
    }

    setDirectionBeforeFirst(direction) {
      this.movementDirection = direction;

      if (!this.parkingApplied && !this.firstArmed) {
        this.applyParking(direction);
      }

      if (this.firstArmed && !this.hasTicked) {
        this.target = this.makeGridTarget(
          this.nextGridValue(this.value, direction),
        );

        this.applyFirstTargetGeometry(this.target, direction);
      }
    }

    setDirectionAfterFirst(direction) {
      this.movementDirection = direction;
      this.target = this.makeGridTarget(
        this.nextGridValue(this.value, direction),
      );

      this.applyNormalTargetGeometry(this.target, direction);
    }

    flashTick(value, intensity = "minor") {
      const tick = this.ticksLayer?.querySelector(
        `.range-tick[data-value="${String(Number(value.toFixed(4)))}"]`,
      );

      if (tick) {
        const className = intensity === "major" ? "is-major-hit" : "is-hit";
        tick.classList.remove("is-hit", "is-major-hit");
        void tick.offsetWidth;
        tick.classList.add(className);

        setTimeout(() => {
          tick.classList.remove("is-hit", "is-major-hit");
        }, 280);
      }

      this.render(this.value, intensity);
    }

    emitTick(gridValue, intensity, pulse) {
      this.root.dispatchEvent(new CustomEvent("haptickit:tick", {
        bubbles: true,
        detail: {
          value: gridValue,
          intensity,
          pulse,
          pattern: this.pattern,
        },
      }));
    }

    handleCrossing() {
      const hitTarget = this.target;
      if (!hitTarget) return;

      this.visualStateOn = !this.visualStateOn;
      this.hasTicked = true;

      if (hitTarget.kind === "accent") {
        this.flashTick(hitTarget.gridValue, "major");
        this.emitTick(hitTarget.gridValue, "major", 2);

        const nextGrid = this.nextGridValue(
          hitTarget.gridValue + this.movementDirection * 0.01,
          this.movementDirection,
        );

        this.scheduleNormalTarget(
          this.makeGridTarget(nextGrid),
          this.movementDirection,
        );

        return;
      }

      const isMajor = this.isMajor(hitTarget.gridValue);
      const weightedMajor = this.pattern === "weighted" && isMajor;
      const intensity = weightedMajor ? "major" : "minor";

      this.flashTick(hitTarget.gridValue, intensity);
      this.emitTick(hitTarget.gridValue, intensity, 1);

      if (weightedMajor) {
        const accent = this.makeAccentTarget(
          hitTarget.gridValue,
          this.movementDirection,
        );

        if (accent) {
          this.scheduleNormalTarget(accent, this.movementDirection);
          return;
        }
      }

      const nextGrid = this.nextGridValue(
        hitTarget.gridValue + this.movementDirection * 0.01,
        this.movementDirection,
      );

      this.scheduleNormalTarget(
        this.makeGridTarget(nextGrid),
        this.movementDirection,
      );
    }

    onPointerDown(event) {
      if (event.pointerType !== "touch" || event.target !== this.driver) return;

      this.activePointerId = event.pointerId;
      this.pointerMoves = 0;
      this.root.classList.add("is-dragging");
      this.latestClientX = event.clientX;
      this.render(this.valueFromClientX(event.clientX));
    }

    onPointerMove(event) {
      if (event.pointerId !== this.activePointerId) return;

      this.pointerMoves += 1;
      this.latestClientX = event.clientX;
      this.render(this.valueFromClientX(event.clientX));
    }

    onPointerEnd(event) {
      if (event.pointerId !== this.activePointerId) return;

      this.activePointerId = null;
      this.root.classList.remove("is-dragging");
    }

    onTouchStart(event) {
      if (event.target !== this.driver || event.touches.length !== 1) return;

      const touch = event.touches[0];
      const rect = this.track.getBoundingClientRect();

      this.activeTouchId = touch.identifier;
      this.gestureStartedAt = performance.now();
      this.startClientX = touch.clientX;
      this.startLocalFull = touch.clientX - rect.left;
      this.latestClientX = touch.clientX;

      this.previousTouchValue = this.valueFromClientX(touch.clientX);
      this.render(this.previousTouchValue);

      this.movementDirection = 0;
      this.parkingApplied = false;
      this.storedStartLocal = null;
      this.firstArmed = false;
      this.hasTicked = false;
      this.visualStateOn = false;
      this.target = null;
      this.pointerMoves = 0;
      this.touchMoves = 0;

      this.resetDriverFull();
      this.setStatus("Pressing · engine warming", false);
      this.root.classList.add("is-dragging");

      clearTimeout(this.armTimer);
      this.armTimer = setTimeout(() => this.armFirstNow(), this.firstArmMs);
    }

    onTouchMove(event) {
      if (this.activeTouchId === null) return;

      const touch = this.touchById(event.touches, this.activeTouchId);
      if (!touch) return;

      this.touchMoves += 1;
      this.latestClientX = touch.clientX;

      const nextValue = this.valueFromClientX(touch.clientX);
      const delta = nextValue - this.previousTouchValue;

      this.render(nextValue);

      if (Math.abs(delta) >= this.directionEpsilon) {
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
        !this.firstArmed
        && this.movementDirection !== 0
        && performance.now() - this.gestureStartedAt >= this.firstArmMs
      ) {
        this.armFirstNow();
      }

      if (
        this.firstArmed
        && this.movementDirection !== 0
        && this.crossed(
          this.previousTouchValue,
          nextValue,
          this.target,
          this.movementDirection,
        )
      ) {
        this.handleCrossing();
      }

      this.previousTouchValue = nextValue;
    }

    onTouchEnd(event) {
      if (this.activeTouchId === null) return;
      if (this.touchById(event.touches, this.activeTouchId)) return;

      this.activeTouchId = null;
      this.firstArmed = false;
      this.root.classList.remove("is-dragging");
      this.setStatus("Hold to engage", false);

      clearTimeout(this.armTimer);
      clearTimeout(this.rearmTimer);

      setTimeout(() => {
        this.driver.checked = false;
        this.resetDriverFull();
      }, 140);
    }

    onResize() {
      this.render(this.value);
    }
  }

  function bindTabs(toast) {
    document.querySelectorAll("[data-tabs]").forEach((tabs) => {
      const items = [...tabs.querySelectorAll("[data-tab]")];
      const panels = [...tabs.querySelectorAll("[data-panel]")];

      items.forEach((item) => {
        const input = item.querySelector(".native-hit");

        input?.addEventListener("change", (event) => {
          const index = Number(item.dataset.tab);

          tabs.style.setProperty("--tab-index", String(index));

          items.forEach((candidate, candidateIndex) => {
            const active = candidateIndex === index;
            candidate.classList.toggle("is-active", active);
            candidate.setAttribute("aria-selected", String(active));
          });

          panels.forEach((panel, panelIndex) => {
            panel.classList.toggle("is-active", panelIndex === index);
          });

          if (event.isTrusted) {
            androidFallback(8);
            toast.show(
              `${item.textContent.trim()} selected`,
              "Trusted native switch interaction.",
            );
          }
        });
      });
    });
  }

  function createParticles(layer) {
    if (!layer) return;

    const colors = ["#a78bfa", "#60a5fa", "#5eead4", "#f9a8d4", "#fde68a"];

    for (let index = 0; index < 18; index += 1) {
      const particle = document.createElement("i");
      const angle = (Math.PI * 2 * index) / 18 + Math.random() * 0.25;
      const distance = 70 + Math.random() * 92;

      particle.className = "particle";
      particle.style.setProperty("--particle-x", `${Math.cos(angle) * distance}px`);
      particle.style.setProperty("--particle-y", `${Math.sin(angle) * distance}px`);
      particle.style.setProperty("--particle-r", `${Math.random() * 260 - 130}deg`);
      particle.style.setProperty("--particle-color", colors[index % colors.length]);
      particle.style.animationDelay = `${Math.random() * 80}ms`;

      layer.appendChild(particle);
      setTimeout(() => particle.remove(), 1000);
    }
  }

  function bindActions(toast) {
    let launchCount = 0;

    document.querySelectorAll("[data-action]").forEach((action) => {
      const input = action.querySelector(".native-hit");

      input?.addEventListener("change", (event) => {
        action.classList.remove("is-fired");
        void action.offsetWidth;
        action.classList.add("is-fired");

        setTimeout(() => action.classList.remove("is-fired"), 520);

        if (event.isTrusted) androidFallback(10);

        const actionName = action.dataset.action;

        if (actionName === "launch") {
          const stage = action.closest("[data-launch-stage]");
          const count = stage?.querySelector("[data-launch-count]");
          const particles = stage?.querySelector("[data-particles]");

          launchCount += 1;
          if (count) count.textContent = String(launchCount);

          stage?.classList.remove("is-fired");
          void stage?.offsetWidth;
          stage?.classList.add("is-fired");

          createParticles(particles);
          setTimeout(() => stage?.classList.remove("is-fired"), 800);

          toast.show("Interaction shipped", "One trusted tap · one native tick.");
          return;
        }

        if (actionName === "hero") {
          document.querySelector("#ranges")?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });

          toast.show("Haptic lab unlocked", "Scroll into the tactile range demos.");
          return;
        }

        toast.show("API marked", "The component surface is ready to extract.");
      });
    });
  }

  function bindToggle(toast) {
    document.querySelectorAll("[data-custom-toggle]").forEach((toggle) => {
      const input = toggle.querySelector(".native-hit");
      const stage = toggle.closest("[data-toggle-stage]");
      const label = stage?.querySelector("[data-toggle-label]");

      const sync = () => {
        const enabled = Boolean(input?.checked);
        toggle.classList.toggle("is-on", enabled);
        stage?.classList.toggle("is-off", !enabled);

        if (label) {
          label.textContent = enabled ? "Haptics enabled" : "Haptics muted";
        }
      };

      input?.addEventListener("change", (event) => {
        sync();

        if (event.isTrusted) {
          androidFallback(input.checked ? 12 : 7);
          toast.show(
            input.checked ? "Haptics enabled" : "Haptics muted",
            "State changed through a native switch.",
          );
        }
      });

      sync();
    });
  }

  function bindRangeEvents(toast) {
    document.addEventListener("haptickit:tick", (event) => {
      const { value, intensity, pulse } = event.detail;

      if (intensity === "major" && pulse === 1) {
        toast.show(
          `Accent at ${Math.round(value)}`,
          "Major mark · double native tick pattern.",
        );
      }
    });
  }

  function initReveal() {
    const elements = document.querySelectorAll(".reveal");

    if (!("IntersectionObserver" in window)) {
      elements.forEach((element) => element.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, {
      threshold: 0.12,
      rootMargin: "0px 0px -40px",
    });

    elements.forEach((element) => observer.observe(element));
  }

  function init() {
    const toast = new Toast(document.querySelector("[data-toast]"));

    const ranges = [...document.querySelectorAll("[data-haptic-range]")]
      .map((element) => new HapticRange(element));

    bindTabs(toast);
    bindActions(toast);
    bindToggle(toast);
    bindRangeEvents(toast);
    initReveal();

    window.HapticKit = Object.freeze({
      Range: HapticRange,
      ranges,
      version: "0.1.0-demo",
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
