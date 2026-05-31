(function () {
    var measurementId = "G-JGG6TEGRN7";
    var host = window.location.hostname;
    var isLocalPreview = window.location.protocol === "file:" ||
        !host ||
        host === "localhost" ||
        host === "127.0.0.1" ||
        host === "::1";

    if (isLocalPreview) {
        return;
    }

    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function () {
        window.dataLayer.push(arguments);
    };

    var script = document.createElement("script");
    script.async = true;
    script.src = "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(measurementId);
    document.head.appendChild(script);

    window.gtag("js", new Date());
    window.gtag("config", measurementId);
})();
