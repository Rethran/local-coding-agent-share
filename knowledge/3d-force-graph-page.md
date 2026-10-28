Static 3D page rules for GRAPH-032 through GRAPH-040:

- Tests are pytest file/string checks and Python helpers. Do not use Selenium, Playwright, or a real browser.
- graph.json is the data-contracts shape. graph-force.json is to_force_graph output `{nodes, links}`.
- CDN script: `<script src="https://unpkg.com/3d-force-graph"></script>` which defines global ForceGraph3D.
- Mount: `ForceGraph3D()(document.getElementById('graph')).graphData(data)`.
- Do not add React, npm, or a bundler.
- Do not call serve_forever from tests. GRAPH-038 tests call static_root() and parse_args() only.
