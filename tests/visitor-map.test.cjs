const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../js/visitor-map.js'), 'utf8');

// A small DOM adapter exercises the production script without network traffic.
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.textContent = ''; }
    appendChild(child) { this.children.push(child); return child; }
    replaceChildren(...children) { this.children = children; this.textContent = ''; }
    setAttribute(name, value) { this.attributes[name] = value; }
    createTHead() { return this.appendChild(new Element('thead')); }
    createTBody() { return this.appendChild(new Element('tbody')); }
    createTFoot() { return this.appendChild(new Element('tfoot')); }
    insertRow() { return this.appendChild(new Element('tr')); }
    insertCell() { return this.appendChild(new Element('td')); }
    find(tag) { return this.children.flatMap(child => [child, ...child.find('*')]).filter(child => tag === '*' || child.tag === tag); }
}

async function run({ counts = {}, location, recentFailure = false, countsFailure = false, storage = new Map(), legacyBrowser = false } = {}) {
    const widget = new Element('div');
    const countries = new Element('div');
    const requests = [];
    const window = {
        location: { hostname: location ? 'zfchenunique.github.io' : '' },
        setTimeout, clearTimeout,
        sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
        fetch: async (url, options) => {
            requests.push({ url, options });
            if (recentFailure && url.endsWith('/recent')) throw new Error('Recent data offline');
            if (countsFailure && url.endsWith('/cities')) throw new Error('Counts offline');
            let data = null;
            if (url.includes('ipwho.is')) data = location;
            else if (url.includes('/increment/')) {
                const { key, by } = JSON.parse(options.body);
                counts[key] = (counts[key] || 0) + by;
                data = { success: true };
            } else if (url.endsWith('/cities')) data = counts;
            else if (options.method === 'POST') data = { success: true };
            return { status: data === null ? 404 : 200, ok: true, json: async () => data };
        }
    };
    vm.runInNewContext(source, {
        window, AbortController, Intl: legacyBrowser ? {} : Intl,
        document: {
            getElementById: id => id === 'visitor-map-widget' ? widget : countries,
            createElement: tag => new Element(tag),
            createElementNS: (_, tag) => new Element(tag)
        }
    });
    await new Promise(resolve => setImmediate(resolve));
    return { widget, countries, requests, counts, storage };
}

function rows(panel, section = 'tbody') {
    return panel.find(section)[0].children.map(row => row.children.map(cell => cell.textContent));
}

test('country totals combine cities and unknown-city visits, ordered by count', async () => {
    const result = await run({ counts: {
        'CN|Beijing|39.91|116.40': 3,
        'CN|Shanghai|31.23|121.47': 2,
        'CN|Unknown||': 1,
        'US|San%20Jose|37.34|-121.89': 4,
        'GB|London|51.50|-0.12': -9,
        'CN|%ZZ||': 40,
        'not-a-location': 80
    } });
    assert.deepEqual(rows(result.countries), [['China', '6', '60.0%'], ['United States', '4', '40.0%']]);
    assert.deepEqual(rows(result.countries, 'tfoot'), [['Total recorded', '10', '100%']]);
    assert.equal(result.widget.find('circle').length, 3);
    assert.equal(result.requests.filter(request => request.options.method === 'POST').length, 0);
});

test('map dot limit does not truncate countries or their visit counts', async () => {
    const counts = Object.fromEntries(Array.from({ length: 105 }, (_, index) => ['US|City' + index + '|30|-100', 2]));
    counts['CN|Beijing|39.91|116.40'] = 1;
    const result = await run({ counts });
    assert.equal(result.widget.find('circle').length, 100);
    assert.deepEqual(rows(result.countries).map(row => row.slice(0, 2)), [['United States', '210'], ['China', '1']]);
    assert.equal(rows(result.countries, 'tfoot')[0][1], '211');
});

test('a country-only visit is counted once without inventing a map location', async () => {
    const options = { location: { success: true, country_code: 'CN', city: null, latitude: 34.8, longitude: 113.7 } };
    const result = await run(options);
    assert.equal(result.counts['CN|Unknown||'], 1);
    assert.deepEqual(rows(result.countries), [['China', '1', '100.0%']]);
    assert.equal(result.widget.find('circle').length, 0);
    const next = await run({ ...options, counts: result.counts, storage: result.storage });
    assert.equal(next.counts['CN|Unknown||'], 1);
    assert.equal(next.requests.filter(request => request.options.method === 'POST').length, 0);
});

test('recent-record failure does not hide country totals or duplicate a recorded visit', async () => {
    const result = await run({ recentFailure: true, location: {
        success: true, country_code: 'CN', city: 'Beijing', latitude: 39.91, longitude: 116.40
    } });
    assert.equal(rows(result.countries)[0][1], '1');
    assert.equal(result.storage.get('zfchen-city-visit-recorded'), '1');
});

test('empty and failed reads remain distinct; older browsers fall back to region codes', async () => {
    assert.equal((await run()).countries.textContent, 'No country visits recorded yet.');
    assert.equal((await run({ countsFailure: true })).countries.textContent, 'Country data temporarily unavailable.');
    const result = await run({ counts: { 'CN|Unknown||': 1 }, legacyBrowser: true });
    assert.equal(rows(result.countries)[0][0], 'CN');
});
