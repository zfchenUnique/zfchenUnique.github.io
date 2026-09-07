# Personal Website

Source for Zhenfang Chen's personal academic website.

## Visitor analytics

The site uses `hits.sh` for the public visitor badge, a locally rendered city-level visitor map, and Google Analytics 4 on the live GitHub Pages domain for private traffic analytics. The public map resolves a visitor's approximate city through `ipwho.is` and saves only anonymous city totals in MantleDB; it does not store raw IP addresses. Visits are counted once per browser tab session.

To view visit time, page path, and approximate country/region/city:

1. Open Google Analytics.
2. Select the GA4 property for `zfchenunique.github.io`.
3. Deploy this repository to GitHub Pages.
4. Use Realtime, User attributes, and Explorations to inspect pages, dates, and geography.

Local previews through `file://`, `localhost`, or `127.0.0.1` do not load GA4 and do not record public-map visits. They can still display the current public city totals for visual testing.
