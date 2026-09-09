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
    var countryStats = document.getElementById("visitor-country-stats");
    var cityRows = document.getElementById("visitor-city-rows");
    var searchInput = document.getElementById("visitor-search");
    var sortInput = document.getElementById("visitor-sort");
    var refreshButton = document.getElementById("visitor-refresh");
    var dataStatus = document.getElementById("visitor-data-status");
    var currentLocations = [];
    var loading = false;
    var hasData = false;

    function countryLabel(code) {
        return code === "XX" ? "Unknown" : countryNames ? countryNames.of(code) : code;
    }

    function renderCityTable() {
        if (!cityRows) return;
        var query = searchInput.value.trim().toLocaleLowerCase();
        var total = currentLocations.reduce(function (sum, location) { return sum + location.count; }, 0);
        var filtered = currentLocations.filter(function (location) {
            return (location.city + " " + location.countryCode + " " + countryLabel(location.countryCode)).toLocaleLowerCase().includes(query);
        }).sort(function (a, b) {
            if (sortInput.value === "city") return a.city.localeCompare(b.city) || b.count - a.count;
            if (sortInput.value === "country") return countryLabel(a.countryCode).localeCompare(countryLabel(b.countryCode)) || b.count - a.count;
            return b.count - a.count || a.city.localeCompare(b.city);
        });
        cityRows.replaceChildren();
        filtered.forEach(function (location) {
            var row = document.createElement("tr");
            [location.city, countryLabel(location.countryCode), location.count.toLocaleString("en"), (location.count / total * 100).toFixed(1) + "%"].forEach(function (value, index) {
                var cell = document.createElement(index === 0 ? "th" : "td");
                if (index === 0) cell.scope = "row";
                cell.textContent = value;
                row.appendChild(cell);
            });
            cityRows.appendChild(row);
        });
        if (!filtered.length) {
            var row = document.createElement("tr");
            var cell = document.createElement("td");
            cell.colSpan = 4;
            cell.textContent = currentLocations.length ? "No matching cities or countries." : "No visits recorded yet.";
            row.appendChild(cell);
            cityRows.appendChild(row);
        }
        var countries = new Set(currentLocations.map(function (location) { return location.countryCode; }));
        document.getElementById("visitor-totals").textContent = total.toLocaleString("en") + " visits · " + currentLocations.length + " location entries · " + countries.size + " countries / regions";
    }

    function renderStatistics(locations, recent) {
        if (!cityRows) return;
        currentLocations = locations;
        hasData = true;
        renderCityTable();
        var latest = document.getElementById("visitor-latest");
        var timestamp = recent && recent.seenAt && new Date(recent.seenAt);
        latest.textContent = recent ? "Latest located visit: " + recent.city + ", " + countryLabel(recent.countryCode) +
            (timestamp && Number.isFinite(timestamp.getTime()) ? " · " + timestamp.toLocaleString(undefined, { timeZoneName: "short" }) : " · Time unavailable") : "Latest visit unavailable.";
        dataStatus.textContent = "Updated " + new Date().toLocaleTimeString() + " · Refreshes every 60 seconds while this panel is visible.";
    }

    if (cityRows) {
        searchInput.addEventListener("input", renderCityTable);
        sortInput.addEventListener("change", renderCityTable);
        refreshButton.addEventListener("click", loadMapData);
        window.setInterval(function () {
            var section = document.getElementById("visitors");
            if (!document.hidden && section && section.open) loadMapData();
        }, 60000);
    }

    var countryNames;
    try {
        countryNames = new Intl.DisplayNames(["en"], { type: "region" });
    } catch (error) {
        // Region codes remain usable on browsers without Intl.DisplayNames.
    }

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
        var mapped = hasCityCoordinates(location.city, location.latitude, location.longitude);
        return [
            cleanLabel(location.country_code, "XX").toUpperCase().slice(0, 2),
            encodeURIComponent(cleanLabel(location.city, "Unknown")),
            mapped ? Number(location.latitude).toFixed(2) : "",
            mapped ? Number(location.longitude).toFixed(2) : ""
        ].join("|");
    }

    function hasCityCoordinates(city, latitude, longitude) {
        return typeof city === "string" && city.trim() !== "" && city !== "Unknown" &&
            latitude !== null && latitude !== "" && longitude !== null && longitude !== "" &&
            Number.isFinite(Number(latitude)) && Math.abs(Number(latitude)) <= 90 &&
            Number.isFinite(Number(longitude)) && Math.abs(Number(longitude)) <= 180;
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

            if (parts.length !== 4 || !/^[A-Z]{2}$/.test(parts[0] || "") ||
                !city || city.length > 60 ||
                !Number.isSafeInteger(count) || count < 1) {
                return null;
            }

            var mapped = hasCityCoordinates(city, parts[2], parts[3]);
            return {
                countryCode: parts[0],
                city: city,
                latitude: mapped ? latitude : null,
                longitude: mapped ? longitude : null,
                count: count
            };
        }).filter(Boolean).sort(function (left, right) {
            return right.count - left.count;
        });
    }

    function renderCountries(locations) {
        if (!countryStats) {
            return;
        }
        countryStats.replaceChildren();
        var counts = Object.create(null);
        var total = 0;
        locations.forEach(function (location) {
            counts[location.countryCode] = (counts[location.countryCode] || 0) + location.count;
            total += location.count;
        });
        if (!total) {
            countryStats.textContent = "No country visits recorded yet.";
            return;
        }

        var table = document.createElement("table");
        table.className = "visitor-country-table";
        var caption = document.createElement("caption");
        caption.textContent = "Visits by country / region";
        table.appendChild(caption);
        var head = table.createTHead().insertRow();
        ["Country / region", "Visits", "Share"].forEach(function (label) {
            var cell = document.createElement("th");
            cell.scope = "col";
            cell.textContent = label;
            head.appendChild(cell);
        });
        var body = table.createTBody();
        Object.keys(counts).sort(function (left, right) {
            return counts[right] - counts[left] || left.localeCompare(right);
        }).forEach(function (code) {
            var row = body.insertRow();
            var name = document.createElement("th");
            name.scope = "row";
            name.textContent = code === "XX" ? "Unknown" : countryNames ? countryNames.of(code) : code;
            row.appendChild(name);
            row.insertCell().textContent = counts[code].toLocaleString("en");
            row.insertCell().textContent = (counts[code] / total * 100).toFixed(1) + "%";
        });
        var footer = table.createTFoot().insertRow();
        var totalLabel = document.createElement("th");
        totalLabel.scope = "row";
        totalLabel.textContent = "Total recorded";
        footer.appendChild(totalLabel);
        footer.insertCell().textContent = total.toLocaleString("en");
        footer.insertCell().textContent = "100%";
        countryStats.appendChild(table);
    }

    function matchesRecent(location, recent) {
        return recent &&
            location.countryCode === recent.countryCode &&
            location.city === recent.city &&
            Math.abs(location.latitude - recent.latitude) < 0.02 &&
            Math.abs(location.longitude - recent.longitude) < 0.02;
    }

    function renderMap(locations, recent) {
        // Country totals include every record, even those without a city or
        // beyond the map's display limit of 100 dots.
        renderCountries(locations);
        var mappedLocations = locations.filter(function (location) {
            return location.latitude !== null && location.longitude !== null;
        });
        overlay.replaceChildren();

        mappedLocations.slice(0, 100).forEach(function (location) {
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

        var chineseCities = mappedLocations.filter(function (location) {
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
                summary.textContent = mappedLocations.length ?
                    mappedLocations.length + " visitor " + (mappedLocations.length === 1 ? "city" : "cities") :
                    "Visits recorded; city locations unavailable.";
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
            !hasCityCoordinates(recent.city, recent.latitude, recent.longitude)) {
            return null;
        }
        return {
            countryCode: recent.countryCode,
            city: cleanLabel(recent.city, "Unknown"),
            latitude: latitude,
            longitude: longitude,
            seenAt: typeof recent.seenAt === "string" ? recent.seenAt : null
        };
    }

    function loadMapData() {
        if (loading) return Promise.resolve();
        loading = true;
        if (refreshButton) refreshButton.disabled = true;
        if (dataStatus) dataStatus.textContent = "Refreshing statistics…";
        return Promise.all([
            fetchJson(citiesUrl),
            fetchJson(recentUrl).catch(function () { return null; })
        ]).then(function (results) {
            var locations = parseLocations(results[0]);
            var recent = normalizedRecent(results[1]);
            renderMap(locations, recent);
            renderStatistics(locations, recent);
        }).catch(function () {
            if (dataStatus) dataStatus.textContent = hasData ? "Refresh failed. Showing the last successful data; try Refresh again." : "Statistics could not load. Try Refresh again.";
            if (cityRows && !hasData) {
                document.getElementById("visitor-totals").textContent = "Statistics unavailable";
            }
            summary.textContent = "City data temporarily unavailable.";
            if (countryStats) {
                countryStats.textContent = "Country data temporarily unavailable.";
            }
            widget.setAttribute("aria-busy", "false");
        }).finally(function () {
            loading = false;
            if (refreshButton) refreshButton.disabled = false;
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
            if (!location || location.success !== true || !/^[A-Z]{2}$/.test(location.country_code || "")) {
                return false;
            }

            var mapped = hasCityCoordinates(location.city, location.latitude, location.longitude);
            var countryCode = cleanLabel(location.country_code, "XX").toUpperCase().slice(0, 2);
            var city = cleanLabel(location.city, "Unknown");
            var recent = {
                countryCode: countryCode,
                city: city,
                latitude: mapped ? Number(Number(location.latitude).toFixed(2)) : null,
                longitude: mapped ? Number(Number(location.longitude).toFixed(2)) : null,
                seenAt: new Date().toISOString()
            };
            var body = JSON.stringify({
                key: locationKey(location),
                by: 1
            });

            return incrementCity(body).then(function () {
                markRecordedThisSession();
                if (!mapped) {
                    return true;
                }
                return fetchJson(recentUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(recent),
                    keepalive: true
                }).catch(function () { return null; }).then(function () { return true; });
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
