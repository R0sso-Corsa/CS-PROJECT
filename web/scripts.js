document.addEventListener("DOMContentLoaded", () => {
    const debugState = {
        maxEntries: 40,
    };

    function createDebugArea() {
        const container = document.createElement("section");
        container.id = "debug-log";

        const heading = document.createElement("h2");
        heading.textContent = "Debug Log";

        const clearButton = document.createElement("button");
        clearButton.type = "button";
        clearButton.textContent = "Clear";

        const list = document.createElement("div");
        list.id = "debug-log-entries";

        container.appendChild(heading);
        container.appendChild(clearButton);
        container.appendChild(list);
        document.body.appendChild(container);

        clearButton.addEventListener("click", () => {
            list.innerHTML = "";
            console.clear();
            console.log("[SignalStack Debug] Debug log cleared.");
        });

        return list;
    }

    const debugList = createDebugArea();

    function appendDebugLine(message, type = "info") {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement("p");
        line.setAttribute("data-type", type);
        line.textContent = `[${timestamp}] ${message}`;
        debugList.prepend(line);

        while (debugList.children.length > debugState.maxEntries) {
            debugList.removeChild(debugList.lastChild);
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
                logDebug(`Ticker quick-fill pressed: ${ticker}`, { ticker }, "action");
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

            const buttonLike = target.closest("button, a, input[type='submit'], input[type='button']");
            if (!(buttonLike instanceof HTMLElement)) {
                return;
            }

            const label = getButtonLabel(buttonLike);
            const href = buttonLike instanceof HTMLAnchorElement ? buttonLike.href : null;
            const form = buttonLike.closest("form");

            logDebug(
                `Element clicked: ${label}`,
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
