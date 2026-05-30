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

    if (placeholder) {
        placeholder.hidden = true;
    }

    window._wau = window._wau || [];
    window._wau_opt = window._wau_opt || {};
    window._wau_opt.target = "_blank";

    var markerId = "visitor_map";
    var configScript = document.createElement("script");
    configScript.id = "_wau" + markerId;
    configScript.text = 'var _wau = _wau || []; _wau.push(["map", "zfchen2026", "' + markerId + '", "280", "140", "night", "cross-pink"]);';
    widget.appendChild(configScript);

    var mapScript = document.createElement("script");
    mapScript.async = true;
    mapScript.src = "https://waust.at/m.js";
    mapScript.onerror = function () {
        if (placeholder) {
            placeholder.hidden = false;
            placeholder.textContent = "Visitor map temporarily unavailable";
        }
    };
    widget.appendChild(mapScript);
})();
