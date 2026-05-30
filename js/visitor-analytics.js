(function () {
    var host = window.location.hostname;
    var isLocalPreview = window.location.protocol === "file:" ||
        !host ||
        host === "localhost" ||
        host === "127.0.0.1" ||
        host === "::1";

    if (isLocalPreview) {
        return;
    }

    window.plausible = window.plausible || function () {
        (window.plausible.q = window.plausible.q || []).push(arguments);
    };
    window.plausible.init = window.plausible.init || function (options) {
        window.plausible.o = options || {};
    };
    window.plausible.init();

    var script = document.createElement("script");
    script.async = true;
    script.src = "https://plausible.io/js/pa-2vz89hXgToIXxmyR12vgh.js";
    document.head.appendChild(script);
})();
