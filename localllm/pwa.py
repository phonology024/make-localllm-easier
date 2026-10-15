"""Install the chat page as an app (#122): web manifest, icon, and a service worker that keeps the page itself for an
offline start. The gateway serves them with the page (public like the page: no data in them).

The service worker caches only the page, the manifest and the icon; the API, chats and status are never cached.
Browsers allow service workers only on https or on localhost, so the app installs (and opens offline) on this PC;
over the LAN on plain http a phone gets an "Add to Home screen" shortcut instead.
"""
from __future__ import annotations

import json

THEME = "#14774c"
ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" rx="112" fill="#14774c"/>
<rect x="96" y="120" width="320" height="272" rx="40" fill="none" stroke="#fff" stroke-width="32"/>
<path d="M168 214l56 46-56 46M264 306h80" fill="none" stroke="#fff" stroke-width="32" stroke-linecap="round" stroke-linejoin="round"/></svg>
"""
MANIFEST = json.dumps({
    "name": "localllm", "short_name": "localllm", "description": "Chat with the AI model running on your own PC",
    "start_url": "/", "scope": "/", "display": "standalone", "background_color": "#141614", "theme_color": THEME,
    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"},
              {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "maskable"}]}, indent=1)
SHELL = ["/", "/manifest.webmanifest", "/icon.svg"]
SW = """// localllm: keep the chat page itself so the app opens without a network; never the API, chats or status.
const SHELL = "localllm-shell-v1", FILES = %s;
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !FILES.includes(url.pathname)) return;
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(url.pathname, copy)); }
    return r;
  }).catch(() => caches.match(url.pathname)));     // network first: the newest page when localllm is running
});
""" % json.dumps(SHELL)

FILES = {"/manifest.webmanifest": ("application/manifest+json", MANIFEST), "/sw.js": ("text/javascript", SW),
         "/icon.svg": ("image/svg+xml", ICON)}
