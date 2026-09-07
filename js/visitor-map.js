(function () {
    var widget = document.getElementById("visitor-map-widget");
    if (!widget) {
        return;
    }

    var isLiveSite = window.location.hostname === "zfchenunique.github.io";
    var storeBase = "https://mantledb.sh/v2/zfchenunique-ghio-visitors-a7c31d";
    var citiesUrl = storeBase + "/cities";
    var recentUrl = storeBase + "/recent";
    var incrementUrl = "https://mantledb.sh/v2/increment/zfchenunique-ghio-visitors-a7c31d/cities";
    var locationUrl = "https://ipwho.is/?fields=success,country_code,city,latitude,longitude";
    var svgNamespace = "http://www.w3.org/2000/svg";

    widget.replaceChildren();
    widget.setAttribute("aria-busy", "true");

    var canvas = document.createElement("div");
    canvas.className = "visitor-map-canvas";

    var baseMap = document.createElement("img");
    baseMap.className = "visitor-map-base";
    baseMap.src = "images/visitor-world-map.svg";
    baseMap.alt = "";
    baseMap.width = 280;
    baseMap.height = 140;

    var overlay = document.createElementNS(svgNamespace, "svg");
    overlay.setAttribute("class", "visitor-map-overlay");
    overlay.setAttribute("viewBox", "0 0 280 140");
    overlay.setAttribute("role", "img");
    overlay.setAttribute("aria-label", "Approximate cities of recent website visitors");

    var summary = document.createElement("div");
    summary.className = "visitor-map-summary";
    summary.textContent = "Loading city-level visits…";

    canvas.appendChild(baseMap);
    canvas.appendChild(overlay);
    widget.appendChild(canvas);
    widget.appendChild(summary);

    function fetchJson(url, options) {
        var controller = new AbortController();
        var timeout = window.setTimeout(function () {
            controller.abort();
        }, 8000);
        var settings = Object.assign({
            cache: "no-store",
            mode: "cors",
            signal: controller.signal
        }, options || {});

        return window.fetch(url, settings).then(function (response) {
            if (response.status === 404) {
                return null;
            }
            if (!response.ok) {
                throw new Error("HTTP " + response.status);
            }
            return response.json();
        }).finally(function () {
            window.clearTimeout(timeout);
        });
    }

    function cleanLabel(value, fallback) {
        if (typeof value !== "string") {
            return fallback;
        }
        var cleaned = value.replace(/[|<>]/g, " ").replace(/\s+/g, " ").trim();
        return cleaned.slice(0, 60) || fallback;
    }

    function locationKey(location) {
        return [
            cleanLabel(location.country_code, "XX").toUpperCase().slice(0, 2),
            encodeURIComponent(cleanLabel(location.city, "Unknown")),
            Number(location.latitude).toFixed(2),
            Number(location.longitude).toFixed(2)
        ].join("|");
    }

    function parseLocations(cityCounts) {
        if (!cityCounts || typeof cityCounts !== "object" || Array.isArray(cityCounts)) {
            return [];
        }

        return Object.keys(cityCounts).map(function (key) {
            var parts = key.split("|");
            var count = Number(cityCounts[key]);
            var latitude = Number(parts[2]);
            var longitude = Number(parts[3]);
            var city;

            try {
                city = decodeURIComponent(parts[1] || "");
            } catch (error) {
                return null;
            }

            if (!/^[A-Z]{2}$/.test(parts[0] || "") ||
                !city || city.length > 60 ||
                !Number.isFinite(latitude) || !Number.isFinite(longitude) ||
                latitude < -90 || latitude > 90 ||
                longitude < -180 || longitude > 180 ||
                !Number.isFinite(count) || count < 1) {
                return null;
            }

            return {
                countryCode: parts[0],
                city: city,
                latitude: latitude,
                longitude: longitude,
                count: Math.min(Math.round(count), 1000000)
            };
        }).filter(Boolean).sort(function (left, right) {
            return right.count - left.count;
        }).slice(0, 100);
    }

    function matchesRecent(location, recent) {
        return recent &&
            location.countryCode === recent.countryCode &&
            location.city === recent.city &&
            Math.abs(location.latitude - recent.latitude) < 0.02 &&
            Math.abs(location.longitude - recent.longitude) < 0.02;
    }

    function renderMap(locations, recent) {
        overlay.replaceChildren();

        locations.forEach(function (location) {
            var circle = document.createElementNS(svgNamespace, "circle");
            var x = (location.longitude + 180) / 360 * 280;
            var y = (90 - location.latitude) / 180 * 140;
            var radius = Math.min(7, 2.3 + Math.log2(location.count + 1));
            var latest = matchesRecent(location, recent);
            var title = document.createElementNS(svgNamespace, "title");
            var visitLabel = location.count === 1 ? "visit" : "visits";

            circle.setAttribute("cx", x.toFixed(2));
            circle.setAttribute("cy", y.toFixed(2));
            circle.setAttribute("r", radius.toFixed(2));
            circle.setAttribute("class", latest ? "visitor-city-dot is-latest" : "visitor-city-dot");
            circle.setAttribute("tabindex", "0");
            title.textContent = location.city + ", " + location.countryCode + " — " + location.count + " " + visitLabel;
            circle.appendChild(title);
            overlay.appendChild(circle);
        });

        var chineseCities = locations.filter(function (location) {
            return location.countryCode === "CN";
        }).slice(0, 5);
        var recentText = recent && recent.city ? "Latest: " + recent.city + ", " + recent.countryCode : "";
        var chinaText = chineseCities.length ? "China: " + chineseCities.map(function (location) {
            return location.city;
        }).join(" · ") : "";

        summary.replaceChildren();
        if (!locations.length) {
            summary.textContent = "Waiting for the first city-level visit.";
        } else {
            if (recentText) {
                var latestLine = document.createElement("strong");
                latestLine.textContent = recentText;
                summary.appendChild(latestLine);
            }
            if (chinaText) {
                var chinaLine = document.createElement("span");
                chinaLine.textContent = chinaText;
                summary.appendChild(chinaLine);
            }
            if (!recentText && !chinaText) {
                summary.textContent = locations.length + " visitor " + (locations.length === 1 ? "city" : "cities");
            }
        }

        widget.setAttribute("aria-busy", "false");
    }

    function normalizedRecent(recent) {
        if (!recent || typeof recent !== "object") {
            return null;
        }
        var latitude = Number(recent.latitude);
        var longitude = Number(recent.longitude);
        if (!recent.city || !/^[A-Z]{2}$/.test(recent.countryCode || "") ||
            !Number.isFinite(latitude) || !Number.isFinite(longitude)) {
            return null;
        }
        return {
            countryCode: recent.countryCode,
            city: cleanLabel(recent.city, "Unknown"),
            latitude: latitude,
            longitude: longitude
        };
    }

    function loadMapData() {
        return Promise.all([
            fetchJson(citiesUrl),
            fetchJson(recentUrl)
        ]).then(function (results) {
            renderMap(parseLocations(results[0]), normalizedRecent(results[1]));
        }).catch(function () {
            summary.textContent = "City data temporarily unavailable.";
            widget.setAttribute("aria-busy", "false");
        });
    }

    function hasRecordedThisSession() {
        try {
            return window.sessionStorage.getItem("zfchen-city-visit-recorded") === "1";
        } catch (error) {
            return false;
        }
    }

    function markRecordedThisSession() {
        try {
            window.sessionStorage.setItem("zfchen-city-visit-recorded", "1");
        } catch (error) {
            // A disabled storage API should not prevent anonymous visit counting.
        }
    }

    function incrementCity(body) {
        var options = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: body,
            keepalive: true
        };

        return fetchJson(incrementUrl, options).then(function (result) {
            if (result) {
                return result;
            }

            // MantleDB removes inactive namespaces. Recreate the aggregate once
            // so the first returning visitor after a quiet period is not lost.
            return fetchJson(citiesUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: "{}",
                keepalive: true
            }).then(function () {
                return fetchJson(incrementUrl, options);
            }).then(function (retryResult) {
                if (!retryResult) {
                    throw new Error("City counter unavailable");
                }
                return retryResult;
            });
        });
    }

    function recordVisit() {
        if (!isLiveSite || hasRecordedThisSession()) {
            return Promise.resolve(false);
        }

        return fetchJson(locationUrl).then(function (location) {
            var latitude = Number(location && location.latitude);
            var longitude = Number(location && location.longitude);
            if (!location || location.success !== true || !location.city ||
                !Number.isFinite(latitude) || !Number.isFinite(longitude)) {
                return false;
            }

            var countryCode = cleanLabel(location.country_code, "XX").toUpperCase().slice(0, 2);
            var city = cleanLabel(location.city, "Unknown");
            var recent = {
                countryCode: countryCode,
                city: city,
                latitude: Number(latitude.toFixed(2)),
                longitude: Number(longitude.toFixed(2)),
                seenAt: new Date().toISOString()
            };
            var body = JSON.stringify({
                key: locationKey(location),
                by: 1
            });

            return Promise.all([
                incrementCity(body),
                fetchJson(recentUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(recent),
                    keepalive: true
                })
            ]).then(function () {
                markRecordedThisSession();
                return true;
            });
        }).catch(function () {
            return false;
        });
    }

    loadMapData().then(function () {
        return recordVisit();
    }).then(function (recorded) {
        if (recorded) {
            return loadMapData();
        }
        return null;
    });
})();
