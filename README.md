# Personal Website

Source for Zhenfang Chen's personal academic website.

## Visitor analytics

The site uses `hits.sh` for the public visitor badge, a locally rendered city-level visitor map, and Google Analytics 4 on the live GitHub Pages domain for private traffic analytics. The public map resolves a visitor's approximate location through `ipwho.is` and saves aggregate location counters and the latest located visit in MantleDB; it does not store raw IP addresses. Visits are counted once per browser tab session.

The country/region table sums all recorded location counters, sorted by visits, with percentages and a total. These counts start with the new tracker on September 7, 2026; they do not include the old map providers' history or use the independent `hits.sh` badge total as their denominator. Country-only visits are retained even if the city is unknown, but are not drawn as city dots. The map displays up to 100 city dots; this limit does not truncate the country totals.

Run the country aggregation and recording regression checks with `node --test tests/visitor-map.test.cjs`. These checks mock the network and never modify live visitor data.

To view visit time, page path, and approximate country/region/city:

1. Open Google Analytics.
2. Select the GA4 property for `zfchenunique.github.io`.
3. Deploy this repository to GitHub Pages.
4. Use Realtime, User attributes, and Explorations to inspect pages, dates, and geography.

Local previews through `file://`, `localhost`, or `127.0.0.1` do not load GA4 and do not record public-map visits. They can still display the current public city totals for visual testing.
