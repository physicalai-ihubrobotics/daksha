"""Operator viewer: watch a session live in the browser (overview camera + Daksha's head camera).

    python scripts/viewer.py --name s1            # then open http://localhost:8090

Read-only: it serves runtime/sessions/<name>/stream/latest.jpg (written by the robot service
while motions run) and a summary of state.json. It never talks to the robot.
"""
import argparse
import http.server
import json
import pathlib

PROJECT = pathlib.Path(__file__).resolve().parents[1]
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Daksha RAX viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#15171a;color:#dde;font:14px system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:24px;flex-wrap:wrap;align-items:baseline}
h1{font-size:16px;margin:0}img{width:100%;max-width:1280px;display:block;margin:0 auto}
.k{color:#8a93a3}</style></head><body>
<header><h1>Daksha · __NAME__</h1><span id="s" class="k">loading…</span></header>
<img id="v" alt="">
<script>
const v=document.getElementById('v'),s=document.getElementById('s');
function tick(){const i=new Image();i.onload=()=>{v.src=i.src};i.src='latest.jpg?t='+Date.now();}
async function st(){try{const r=await fetch('state.json?t='+Date.now());const j=await r.json();
if(!j.launched){s.textContent='Session prepared but not launched: run  harness.launcher launch --name '+j.name;return;}
s.textContent=`episodes ${j.episodes_started}/${j.budget} · current ${j.current??'none'} · finished results ${JSON.stringify(j.results)}`+(j.has_frame?'':' · waiting for the first frame (the agent has not called start_episode yet)');}catch(e){s.textContent='viewer error: '+e;}}
setInterval(tick,200);setInterval(st,2000);tick();st();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to serve on every network interface')
    args = ap.parse_args()
    root = PROJECT/'runtime'/'sessions'/args.name

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                return self._send(PAGE.replace('__NAME__', args.name).encode(), 'text/html')
            if path == '/latest.jpg':
                f = root/'stream'/'latest.jpg'
                if f.exists():
                    return self._send(f.read_bytes(), 'image/jpeg')
            if path == '/state.json':
                if not (root/'state.json').exists():
                    return self._send(json.dumps({'launched': False, 'name': args.name}).encode(), 'application/json')
                try:
                    st = json.loads((root/'state.json').read_text())
                    cfg = json.loads((root/'configuration.json').read_text())
                    cur = st.get('current')
                    return self._send(json.dumps({'launched': True, 'has_frame': (root/'stream'/'latest.jpg').exists(),
                                                  'episodes_started': st['episodes_started'], 'budget': cfg['episodes'],
                                                  'current': None if cur is None else cur['index'] + 1,
                                                  'results': [h.get('final') for h in st['history']]}).encode(),
                                      'application/json')
                except (OSError, ValueError, KeyError):
                    pass
            self.send_error(404)

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Viewer for {args.name}: http://{args.host}:{args.port}  (0.0.0.0 = use this machine IP)')
    server.serve_forever()


if __name__ == '__main__':
    main()
