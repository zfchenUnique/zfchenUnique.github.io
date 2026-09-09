# Personal Website

Source for Zhenfang Chen's personal academic website.

## Visitor analytics

The site uses `hits.sh` for the public visitor badge, a locally rendered city-level visitor map, and Google Analytics 4 on the live GitHub Pages domain for private traffic analytics. The public map resolves a visitor's approximate location through `ipwho.is` and saves aggregate location counters and the latest located visit in MantleDB; it does not store raw IP addresses. Visits are counted once per browser tab session.

The country/region table sums all recorded location counters, sorted by visits, with percentages and a total. These counts start with the new tracker on September 7, 2026; they do not include the old map providers' history or use the independent `hits.sh` badge total as their denominator. Country-only visits are retained even if the city is unknown, but are not drawn as city dots. The map displays up to 100 city dots; this limit does not truncate the country totals.

Run the country aggregation and recording regression checks with `node --test tests/visitor-map.test.cjs`. These checks mock the network and never modify live visitor data.

To view visit time, page path, and approximate country/region/city:

1. Open Google Analytics.
2. Select the GA4 property for `zfchenunique.github.io`.
3. Use Realtime for recent activity. For historical traffic, open Explore → Free form.
4. Import the dimensions Date + hour, City, and Page path and screen class, plus the Views metric. Add the dimensions to Rows and Views to Values, select a date range, and filter by city or page.

Deploy repository changes to GitHub Pages to update the public panel. The redesigned panel provides a larger map, a clearly labeled page-hit counter, and a private report entry point. It does not fetch GA4 data or expose private analytics to public visitors. The responsive map preserves the existing city collection and country totals.

The GA4 measurement ID in the source is a collection identifier, not a reporting credential or numeric property ID. Access to the matching Google Analytics property is needed to read real reports. Date + hour uses the property's time zone. Reports are aggregated, may be delayed or privacy-thresholded, and cannot reconstruct individual IP visits. GA4 does not store raw IP addresses. IP-derived city locations are approximate.

The hits.sh badge counts page hits rather than unique people; its totals need not match GA4 or the city tracker. Individual visits with raw IP addresses would require a separate authenticated backend and a new collection integration; this static website does not currently implement that service.

Local previews through `file://`, `localhost`, or `127.0.0.1` do not load GA4 and do not record public-map visits. They can still display the current public city totals for visual testing.

## Automatic city statistics table

The Visitors panel reads the same city aggregates as the map and displays all location entries in a searchable, sortable table. Shares always use all recorded visits as the denominator, including while filtering. The latest located visit displays the stored timestamp in the viewer's local time zone. The recent visit table stores timestamped tab sessions separately; raw IPs are not stored.

A manual Refresh button and a 60-second refresh while the panel is open and the page is visible only read existing data. They do not count additional visits. Failed refreshes keep the previous city table with a stale-data message.

## Recent visit timestamps

Starting with this update, a successfully located and counted tab session also writes arrival time (ISO UTC from the visitor device), city, country code and pathname to `visit-history`. Query strings, fragments and raw IPs are excluded. The public table shows the latest 100 records in the viewer’s local time zone. Previous aggregate counts cannot be backfilled. Existing tab sessions already counted before this update begin recording on their next new tab session.

Writes use UUID fields and merge patches so simultaneous new records do not replace each other. Each write removes observed entries beyond the newest 99 before adding itself; concurrent writes may temporarily leave more than 100 stored records until another write prunes them. City totals remain independent of history retention and history write failures. This is public, browser-reported telemetry, not an authenticated audit log. MantleDB free storage limits and network failures can prevent writes; a failed write is displayed in the history status. The unclaimed namespace may expire after 30 days without writes.
