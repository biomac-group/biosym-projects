"""
Live view of an IPOPT log in the browser.

    python scripts/serve_log.py LOGFILE [--port 8765]

Serves a page at http://localhost:PORT with the objective (y range: its lowest value so far to that plus 3), the primal and the dual infeasibility over the iterations
(parsed from the iteration lines of the log, refreshed every 5 s) and the last lines of the log.
"""
import argparse
import http.server
from pathlib import Path

PAGE = """<!doctype html><meta charset=utf-8><title>IPOPT progress</title>
<style>body{font:14px system-ui;margin:16px;max-width:1100px}h1{font-size:18px}canvas{width:100%;height:230px;border:1px solid #ccc;margin:6px 0}
pre{background:#f4f4f4;padding:8px;overflow:auto;max-height:260px;font-size:12px}#s{color:#333;margin:4px 0}</style>
<h1 id=t></h1><div id=s></div>
<b>Objective</b> (window: best value so far to +3)<canvas id=a></canvas><b>Infeasibilities (log10)</b> <span style="color:#d55e00">primal</span> / <span style="color:#0072b2">dual</span><canvas id=b></canvas>
<b>Log tail</b><pre id=l></pre>
<script>
const TITLE=__TITLE__;document.getElementById('t').textContent=TITLE;
function draw(c,series,win){const g=c.getContext('2d'),W=c.width=c.clientWidth*2,H=c.height=c.clientHeight*2;g.clearRect(0,0,W,H);
 const all=series.flatMap(s=>s.y.filter(Number.isFinite));if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(win){hi=lo+win}if(lo==hi){lo-=1;hi+=1}
 const n=Math.max(...series.map(s=>s.y.length))-1||1;g.font='20px system-ui';g.fillStyle='#555';g.fillText(hi.toPrecision(3),4,22);g.fillText(lo.toPrecision(3),4,H-6);
 for(const s of series){g.strokeStyle=s.c;g.lineWidth=2.5;g.beginPath();s.y.forEach((v,i)=>{const x=60+(W-70)*i/n,y=Math.min(H-20,Math.max(20,H-20-(H-40)*(v-lo)/(hi-lo)));i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}}
async function tick(){const r=await fetch('/log?'+Date.now());const txt=await r.text();const rows=[];
 for(const ln of txt.split('\\n')){const m=ln.match(/^\\s*(\\d+)r?\\s+(-?\\d\\.\\d+e[+-]\\d+)\\s+(\\d\\.\\d+e[+-]\\d+)\\s+(\\d\\.\\d+e[+-]\\d+)/);if(m)rows.push([+m[1],+m[2],+m[3],+m[4]])}
 if(rows.length){const l=rows[rows.length-1];document.getElementById('s').textContent=`iteration ${l[0]}   objective ${l[1].toPrecision(6)}   primal infeasibility ${l[2].toExponential(2)}   dual infeasibility ${l[3].toExponential(2)}   (${new Date().toLocaleTimeString()})`;
  draw(document.getElementById('a'),[{y:rows.map(r=>r[1]),c:'#222'}],3);
  draw(document.getElementById('b'),[{y:rows.map(r=>Math.log10(Math.max(r[2],1e-16))),c:'#d55e00'},{y:rows.map(r=>Math.log10(Math.max(r[3],1e-16))),c:'#0072b2'}])}
 const lines=txt.trim().split('\\n');document.getElementById('l').textContent=lines.slice(-14).join('\\n')}
tick();setInterval(tick,5000);
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    log = Path(args.log).resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/log"):
                body, kind = log.read_text(errors="replace").encode(), "text/plain; charset=utf-8"
            else:
                body, kind = PAGE.replace("__TITLE__", repr(str(log.name))).encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"serving {log} at http://localhost:{args.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
