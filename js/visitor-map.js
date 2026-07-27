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
        placeholder.remove();
    }

    var mapScript = document.createElement("script");
    mapScript.id = "mapmyvisitors";
    mapScript.src = "https://mapmyvisitors.com/map.js?d=xI7eD8AslLuWPqdkVrRicZbXeV1PgiX-O4BbvZkMNbc&cl=ffffff&w=a";
    mapScript.onerror = function () {
        var fallback = document.createElement("span");
        fallback.className = "visitor-map-placeholder";
        fallback.textContent = "Visitor map temporarily unavailable";
        widget.appendChild(fallback);
    };
    widget.appendChild(mapScript);
})();
