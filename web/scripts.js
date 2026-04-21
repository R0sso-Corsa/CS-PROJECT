document.addEventListener("DOMContentLoaded", () => {
    const debugState = {
        maxEntries: 40,
    };

    function createDebugPanel() {
        const panel = document.createElement("aside");
        panel.className = "debug-panel";
        panel.innerHTML = `
            <div class="debug-panel__header">
                <strong>Button Debug</strong>
                <button type="button" class="debug-panel__clear">Clear</button>
            </div>
            <div class="debug-panel__body"></div>
        `;

        document.body.appendChild(panel);

        const body = panel.querySelector(".debug-panel__body");
        const clearButton = panel.querySelector(".debug-panel__clear");

        clearButton?.addEventListener("click", () => {
            if (body) {
                body.innerHTML = "";
            }
            console.clear();
            console.log("[SignalStack Debug] Debug panel cleared.");
        });

        return {
            panel,
            body,
        };
    }

    const debugPanel = createDebugPanel();

    function appendDebugLine(message, type = "info") {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement("div");
        line.className = `debug-line debug-line--${type}`;
        line.textContent = `[${timestamp}] ${message}`;

        debugPanel.body?.prepend(line);

        while (debugPanel.body && debugPanel.body.children.length > debugState.maxEntries) {
            debugPanel.body.removeChild(debugPanel.body.lastChild);
        }
    }

    function logDebug(message, payload = null, type = "info") {
        appendDebugLine(message, type);
        if (payload !== null) {
            console.log(`[SignalStack Debug] ${message}`, payload);
        } else {
            console.log(`[SignalStack Debug] ${message}`);
        }
    }

    function getButtonLabel(element) {
        const explicit = element.getAttribute("data-debug-label");
        if (explicit) {
            return explicit;
        }

        const text = (element.textContent || "").trim().replace(/\s+/g, " ");
        if (text) {
            return text;
        }

        return element.getAttribute("name") || element.getAttribute("id") || element.tagName.toLowerCase();
    }

    function sanitizeFormData(form) {
        const formData = new FormData(form);
        const entries = {};

        for (const [key, value] of formData.entries()) {
            if (/pass(word)?/i.test(key)) {
                entries[key] = `[hidden:${String(value).length}]`;
            } else {
                entries[key] = String(value);
            }
        }

        return entries;
    }

    const searchInput = document.querySelector('input[name="ticker"]');
    const fillButtons = document.querySelectorAll("[data-fill-ticker]");

    fillButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (searchInput) {
                const ticker = button.getAttribute("data-fill-ticker") || "";
                searchInput.value = ticker;
                searchInput.focus();
                logDebug(`Ticker chip pressed: ${ticker}`, { ticker }, "action");
            }
        });
    });

    document.addEventListener(
        "click",
        (event) => {
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }

            const buttonLike = target.closest("button, a.button, input[type='submit'], input[type='button']");
            if (!(buttonLike instanceof HTMLElement)) {
                return;
            }

            const label = getButtonLabel(buttonLike);
            const href = buttonLike instanceof HTMLAnchorElement ? buttonLike.href : null;
            const form = buttonLike.closest("form");

            logDebug(
                `Button clicked: ${label}`,
                {
                    tag: buttonLike.tagName.toLowerCase(),
                    href,
                    formAction: form?.getAttribute("action") || null,
                    formMethod: form?.getAttribute("method") || null,
                },
                "action"
            );
        },
        true
    );

    document.addEventListener(
        "submit",
        (event) => {
            const form = event.target;
            if (!(form instanceof HTMLFormElement)) {
                return;
            }

            const action = form.getAttribute("action") || window.location.pathname;
            const method = (form.getAttribute("method") || "get").toUpperCase();
            const payload = sanitizeFormData(form);

            logDebug(
                `Form submitted: ${method} ${action}`,
                {
                    method,
                    action,
                    payload,
                },
                "submit"
            );
        },
        true
    );

    window.signalStackDebug = {
        log: logDebug,
    };

    logDebug("Frontend debug instrumentation loaded.", { path: window.location.pathname }, "system");
});
