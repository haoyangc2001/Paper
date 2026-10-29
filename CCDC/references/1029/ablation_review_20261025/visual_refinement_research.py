"""Fetch focused figure-design references; never change paper or plot data."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests
from bs4 import BeautifulSoup

DEST=Path(__file__).resolve().parent
JOBS=[
 ('estimation','https://www.nature.com/articles/s41592-019-0470-3'),
 ('estimation_figure','https://www.nature.com/articles/s41592-019-0470-3/figures/1'),
 ('estimation_pmc','https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41592-019-0470-3&format=json'),
 ('better_figures','https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833'),
 ('safety_gymnasium','https://arxiv.org/abs/2310.12567'),
 ('rliable','https://arxiv.org/abs/2108.13264'),
]

def fetch(job):
    key,url=job
    result={'key':key,'url':url}
    try:
        response=requests.get(url,timeout=25)
        result['status']=response.status_code
        if key.endswith('_pmc'):
            result['body']=response.json()
        else:
            soup=BeautifulSoup(response.text,'html.parser')
            for tag in soup.select('script,style,nav,header,footer'):tag.decompose()
            main=soup.select_one('main') or soup
            result['title']=soup.title.get_text() if soup.title else None
            result['text']=main.get_text(' ',strip=True)
            result['figures']=[{'src':i.get('src'),'alt':i.get('alt')} for i in soup.select('figure img')]
            result['pdf_links']=[a.get('href') for a in soup.select('a[href]') if 'pdf' in a.get('href','').lower()][:15]
    except Exception as error:result['error']=str(error)
    return result

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(fetch,JOBS))
    (DEST/'visual_refinement_web.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in results:
        text=r.get('text','')
        excerpts=[]
        for term in ['Fig. 1','Figure 1','Rule 2','Rule 5','Rule 6','Rule 7','direct','paired']:
            p=text.find(term)
            if p>=0:excerpts.append(text[max(0,p-100):p+1900])
        print(json.dumps({k:v for k,v in r.items() if k!='text'}|{'text_length':len(text),'excerpts':excerpts},ensure_ascii=False))
