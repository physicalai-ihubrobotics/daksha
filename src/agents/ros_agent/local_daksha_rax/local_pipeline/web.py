"""Browser UI for the local pipeline: live sim, VLM overlay, chat and a Stop button.

GET  /             page
GET  /latest.jpg   live sim stream [overview | head camera] (written by the simulator)
GET  /vlm.jpg      last head-camera frame with the VLM's detections drawn on it
GET  /state.json   agent state, chat, log
POST /command      {"text": "..."}  instruction or question (motion commands barge in)
POST /stop         interrupt now
"""
import http.server
import json

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Local Daksha RAX</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c313a;--fg:#e2e5ea;--mute:#8d96a5;--acc:#5fb3ff;--stop:#e5484d;--ok:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0}.pill{padding:2px 9px;border-radius:99px;background:var(--panel);color:var(--mute);font-size:12px}
.pill.busy{color:#111;background:#f0b429}.pill.ok{color:#111;background:var(--ok)}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:12px;padding:12px 16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:0 0 8px}
img{width:100%;display:block;border-radius:6px;background:#000}
#chat{height:300px;overflow:auto;display:flex;flex-direction:column;gap:6px}
.m{padding:6px 9px;border-radius:8px;max-width:92%;white-space:pre-wrap}.m.operator{align-self:flex-end;background:#24476b}
.m.daksha{background:#262a31}.m small{display:block;color:var(--mute);font-size:11px}
form{display:flex;gap:6px;margin-top:8px}input{flex:1;min-width:0;padding:9px;border-radius:8px;border:1px solid var(--line);background:#101216;color:var(--fg)}
button{padding:9px 14px;border:0;border-radius:8px;background:var(--acc);color:#071321;font-weight:600;cursor:pointer}
button.stop{background:var(--stop);color:#fff;width:100%;font-size:16px;padding:12px;margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.chips button{background:#2a2f38;color:var(--fg);font-weight:400;padding:5px 9px;font-size:12px}
pre{margin:0;max-height:220px;overflow:auto;font:12px/1.4 ui-monospace,monospace;color:#b9c0cc;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:12px}td{padding:3px 4px;border-bottom:1px solid var(--line)}
</style></head><body>
<header><h1>Daksha · local VLM pipeline</h1><span id="busy" class="pill">idle</span>
<span id="task" class="pill"></span><span id="vlm" class="pill"></span></header>
<main>
 <div style="display:grid;gap:12px;min-width:0">
  <div class="card"><h2>Simulation · overview | head camera (live)</h2><img id="live" alt="live simulation"></div>
  <div class="card"><h2 id="vlmh">VLM perception on the head camera</h2><img id="vlmimg" alt="no detections yet: type look"></div>
 </div>
 <div style="display:grid;gap:12px;align-content:start;min-width:0">
  <div class="card"><button class="stop" id="stop">■ STOP / interrupt</button>
   <h2>Talk to Daksha</h2><div id="chat"></div>
   <form id="f"><input id="t" placeholder="say anything: pick up the yellow fruit · move the apple to the left · what is in the bowl?" autocomplete="off"><button>Send</button></form>
   <div class="chips"></div></div>
  <div class="card"><h2>Detections (robot frame, m)</h2><table id="det"></table></div>
  <div class="card"><h2>Log</h2><pre id="log"></pre></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const chips=['what do you see?','grab the red one and drop it in the white dish','put the banana next to the orange','hand me the orange','lift your left hand a little','tidy up the table','then go home','reset'];
document.querySelector('.chips').innerHTML=chips.map(c=>`<button type=button>${c}</button>`).join('');
document.querySelectorAll('.chips button').forEach(b=>b.onclick=()=>send(b.textContent));
async function send(text){await fetch('command',{method:'POST',body:JSON.stringify({text})});poll();}
$('f').onsubmit=e=>{e.preventDefault();const v=$('t').value;$('t').value='';if(v.trim())send(v);};
$('stop').onclick=async()=>{await fetch('stop',{method:'POST'});poll();};
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('stop').click();});
function frame(id,src){const i=new Image();i.onload=()=>{$(id).src=i.src};i.src=src+'?t='+Date.now();}
let lastChat=0;
async function poll(){try{const s=await (await fetch('state.json?t='+Date.now())).json();
 const held=Object.entries(s.held||{}).filter(([a,o])=>o).map(([a,o])=>`${a} hand: ${o}`).join(', ');
 $('busy').textContent=(s.thinking?'understanding… · ':'')+(s.busy?('running: '+s.current+(s.queued?` (+${s.queued} queued)`:'')):'idle')+(held?' · '+held:'');$('busy').className='pill'+(s.busy||s.thinking?' busy':'');
 const ts=s.task_state;$('task').textContent=`${s.task} · seed ${s.seed}`+(ts?` · inside: ${JSON.stringify(ts.inside)}${ts.task_complete?' · COMPLETE':''}`:'');
 $('task').className='pill'+(ts&&ts.task_complete?' ok':'');
 $('vlmh').textContent=s.model+' perception on the head camera';$('vlm').textContent=`${s.model}`+(s.vlm&&s.vlm.seconds?` · last ${s.vlm.seconds}s`:'')+` · interrupts ${s.interrupts}`;
 if(s.chat.length!==lastChat){lastChat=s.chat.length;$('chat').innerHTML=s.chat.map(m=>`<div class="m ${m.who}"><small>${m.who} · ${m.t}</small>${m.text.replace(/</g,'&lt;')}</div>`).join('');$('chat').scrollTop=1e9;}
 $('det').innerHTML=Object.entries(s.detections||{}).map(([k,v])=>`<tr><td>${k}</td><td>${v.centre.map(x=>x.toFixed(3)).join(', ')}</td><td>${v.source}</td></tr>`).join('')||'<tr><td>none yet</td></tr>';
 const lg=$('log');const atEnd=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-4;lg.textContent=s.logs.join('\n');if(atEnd)lg.scrollTop=1e9;
}catch(e){$('busy').textContent='UI lost the pipeline: '+e;}}
setInterval(()=>frame('live','latest.jpg'),150);setInterval(()=>frame('vlmimg','vlm.jpg'),1000);setInterval(poll,700);poll();
</script></body></html>"""


def serve(agent, host='127.0.0.1', port=8092):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                return self._send(PAGE.encode(), 'text/html; charset=utf-8')
            if path in ('/latest.jpg', '/vlm.jpg'):
                f = agent.stream_dir/path[1:]
                try:
                    return self._send(f.read_bytes(), 'image/jpeg')
                except OSError:
                    return self.send_error(404)
            if path == '/state.json':
                return self._send(json.dumps(agent.snapshot(), default=str).encode(), 'application/json')
            self.send_error(404)

        def do_POST(self):
            path = self.path.split('?')[0]
            n = min(int(self.headers.get('Content-Length') or 0), 65536)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                body = {}
            if path == '/command':
                agent.submit(str(body.get('text', ''))[:500])
            elif path == '/stop':
                agent.stop()
            else:
                return self.send_error(404)
            self._send(b'{"ok":true}', 'application/json')

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
