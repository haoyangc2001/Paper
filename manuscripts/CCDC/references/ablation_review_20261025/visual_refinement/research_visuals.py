"""Archive literature evidence for figure refinement; do not change our chart."""
from pathlib import Path
import json
import re
import sys
import requests
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    paper=HERE/'robomimic.pdf'
    if not paper.exists():
        response=requests.get('https://arxiv.org/pdf/2108.03298',timeout=45)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError('Not a PDF')
        paper.write_bytes(response.content)
    with fitz.open(paper) as doc:
        hits=[]
        for i,page in enumerate(doc):
            text=page.get_text()
            if re.search('ablation|Figure [45678]:',text,re.I):
                hits.append({'page':i+1,'text':text})
        (HERE/'robomimic_pages.json').write_text(json.dumps(hits,ensure_ascii=False,indent=2),encoding='utf-8')
        print('robomimic pages',len(doc),'relevant',[p['page'] for p in hits])
        for hit in hits[:10]:
            text=hit['text']
            print('PAGE',hit['page'])
            for m in re.finditer(r'Figure \d+:|ablation',text,re.I):
                print(re.sub(r'\s+',' ',text[max(0,m.start()-60):m.start()+850]))
    for key,file,page in [('safety_gymnasium','CCDC/references/metrics_review_20261022/papers/safety_gymnasium.pdf',9),('rliable','CCDC/references/metrics_review_20261022/papers/rliable.pdf',8),('confined','CCDC/references/figure_review_20261020/papers/confined.pdf',5)]:
        with fitz.open(ROOT/file) as doc:
            text=doc[page-1].get_text()
            (HERE/(key+'_page.txt')).write_text(text,encoding='utf-8')
            print(key,'PAGE',page,text)

if __name__=='__main__':
    main()
