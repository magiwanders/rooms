#!/usr/bin/env python3
"""Local-only static server for web/ that tells the browser not to cache (dev builds change often)."""
import functools, http.server, os, sys

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

port = int(os.environ.get('PORT', 8642))
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(NoCache, directory=root)).serve_forever()
