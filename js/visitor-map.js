(function () {
    var widget = document.getElementById("visitor-map-widget");
    if (!widget) {
        return;
    }

    var placeholder = widget.querySelector(".visitor-map-placeholder");
    var isLiveSite = window.location.hostname === "zfchenunique.github.io";

    if (!isLiveSite) {
        return;
    }

    widget.setAttribute("aria-busy", "true");

    function showUnavailableMessage() {
        if (!placeholder) {
            placeholder = document.createElement("span");
            placeholder.className = "visitor-map-placeholder";
            widget.insertBefore(placeholder, widget.firstChild);
        }

        placeholder.textContent = "Visitor map temporarily unavailable";
        widget.setAttribute("aria-busy", "false");
    }

    function hasRenderedMap() {
        return Boolean(widget.querySelector('img[src*="widgets.amung.us/mapbacks/"]'));
    }

    var renderTimeout;
    var observer = new MutationObserver(function () {
        if (!hasRenderedMap()) {
            return;
        }

        window.clearTimeout(renderTimeout);
        observer.disconnect();
        if (placeholder) {
            placeholder.remove();
        }
        widget.setAttribute("aria-busy", "false");
    });
    observer.observe(widget, { childList: true, subtree: true });

    window._wau = window._wau || [];
    window._wau_opt = window._wau_opt || {};
    window._wau_opt.target = "_blank";
    window._wau_opt.fd = false;

    var markerId = "visitor_map";
    var configScript = document.createElement("script");
    configScript.id = "_wau" + markerId;
    configScript.text = 'var _wau = _wau || []; _wau.push(["map", "zfchen2026", "' + markerId + '", "280", "140", "night", "cross-pink"]);';
    widget.appendChild(configScript);

    var mapScript = document.createElement("script");
    mapScript.async = true;
    mapScript.src = "https://waust.at/m.js";
    mapScript.onerror = showUnavailableMessage;
    widget.appendChild(mapScript);

    renderTimeout = window.setTimeout(function () {
        if (!hasRenderedMap()) {
            showUnavailableMessage();
        }
    }, 10000);

    if (hasRenderedMap()) {
        window.clearTimeout(renderTimeout);
        observer.disconnect();
        if (placeholder) {
            placeholder.remove();
        }
        widget.setAttribute("aria-busy", "false");
    }
})();
