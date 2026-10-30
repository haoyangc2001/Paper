"""Write or verify the self-contained source and delivery inventories."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(paths):return [{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(paths)]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');a=ap.parse_args()
source_path=ROOT/'source_manifest.json';delivery_path=ROOT/'output/final_manifest.json'
if a.write:
    # Existing final graphics must retain the already approved hashes.
    previous=read(delivery_path)
    for row in previous['files']:
        path=ROOT/row.get('target',row.get('path'))
        assert path.is_file() and sha(path)==row['sha256'],str(path)
    inputs=[p for folder in ['assets','config','data','preview'] for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(ROOT/'preview/build')]
    inputs+=list((ROOT/'scripts').glob('render_*_remote.py'))
    write(source_path,{'schema':'local-relative-sha256-v1','data_status':'illustrative_assumed',
        'render_status':'static_mujoco_egl','files':records(inputs)})
    sources=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not p.is_relative_to(ROOT/'output') and not p.is_relative_to(ROOT/'preview/build')]
    finals=[p for p in (ROOT/'output/final').rglob('*') if p.is_file()]
    write(delivery_path,{'version':'v17-03','packaging':'self_contained_cleanup_20261022',
        'files':records(finals),'sources':records(sources),'pass':True})
manifest=read(delivery_path)
for row in manifest['files']+manifest['sources']:
    assert (ROOT/row['path']).is_file() and sha(ROOT/row['path'])==row['sha256'],row['path']
print(json.dumps({'pass':True,'final_files':len(manifest['files']),'sources':len(manifest['sources'])}))
