"""Read public conference pages and retain reproducible source snapshots."""
import concurrent.futures
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
ROOT.joinpath('sources').mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')

def fetch(url):
    key = re.sub(r'[^a-zA-Z0-9_-]+', '_', url.removeprefix('https://').removeprefix('http://'))[:150]
    target = ROOT / 'sources' / key
    if target.with_suffix('.json').exists():
        record = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
        if record.get('ok'):
            return record
    proc = subprocess.run(['curl.exe', '-L', '--max-time', '22', '-sS', '-w', '\n%{http_code}\n%{url_effective}', url], capture_output=True)
    pieces = proc.stdout.rsplit(b'\n', 2)
    body = pieces[0] if len(pieces) == 3 else proc.stdout
    status = pieces[1].decode(errors='replace') if len(pieces) == 3 else ''
    soup = BeautifulSoup(body, 'html.parser')
    for tag in soup(['script','style','noscript']):
        tag.decompose()
    lines = [re.sub(r'\s+', ' ', s).strip() for s in soup.get_text('\n', strip=True).splitlines()]
    text = '\n'.join(s for s in lines if s)
    links = [{'label':a.get_text(' ',strip=True),'url':urljoin(url,a['href'])} for a in soup.find_all('a',href=True)]
    record = {'url':url,'effective_url':pieces[-1].decode(errors='replace') if len(pieces)==3 else url,'status':status,'curl_exit':proc.returncode,'ok':proc.returncode==0 and status=='200' and len(text)>100,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(body).hexdigest(),'text_file':target.with_suffix('.txt').name,'links':links,'error':proc.stderr.decode(errors='replace')[:400]}
    target.with_suffix('.html').write_bytes(body)
    target.with_suffix('.txt').write_text(text,encoding='utf-8')
    target.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

def show(record):
    text = (ROOT/'sources'/record['text_file']).read_text(encoding='utf-8')
    lines = text.splitlines()
    regex = re.compile(r'2027|2026|simultan|elsewhere|under (?:review|consideration)|duplicate|multiple submission|previously|not.*publish|original|notification|acceptance|submission deadline|China|Hong Kong|Macau|中国|一稿多投|重复投稿',re.I)
    matched = [i for i,l in enumerate(lines) if regex.search(l)]
    indexes = sorted({j for i in matched for j in range(max(0,i-1),min(len(lines),i+3))})
    excerpts = '\n'.join(lines[i] for i in indexes)
    selected = [l for l in record['links'] if re.search(r'submi|date|author|ethic|polic|cfp|robot|control|2027',l['label']+' '+l['url'],re.I)]
    return {'url':record['url'],'status':record['status'],'ok':record['ok'],'excerpts':excerpts[:17000],'links':selected[:45]}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for record in executor.map(fetch,sys.argv[1:]):
            print(json.dumps(show(record),ensure_ascii=False))
