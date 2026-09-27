"""Validate design contracts, content registration and local website links."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ERRORS = []

def require(condition, message):
    if not condition: ERRORS.append(message)

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids=set(); self.duplicates=[]; self.refs=[]; self.styles=0; self.inline_scripts=0; self.body=False; self.h1=0
        self.feed(text)
    def handle_starttag(self, tag, pairs):
        attrs=dict(pairs)
        if tag=='body': self.body=True
        if tag=='h1': self.h1+=1
        if 'id' in attrs:
            if attrs['id'] in self.ids: self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        if tag=='style': self.styles+=1
        if tag=='script' and not attrs.get('src'): self.inline_scripts+=1
        if tag in ['a','link'] and attrs.get('href'): self.refs.append(attrs['href'])
        if tag in ['script','img','source'] and attrs.get('src'): self.refs.append(attrs['src'])

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/build_site.py'),'--check'],check=True)
    data=json.loads((ROOT/'content-index.json').read_text(encoding='utf-8'))
    ids=set(); sources=set(); topics={item['id'] for item in data['topics']}
    sections={item['id'] for item in data['sections']}
    require(len(sections)==len(data['sections']),'Duplicate section id')
    collection_sections={item['id']:item['section'] for item in data['collections']}
    for cid,section in collection_sections.items():
        require(section in sections,'Unknown collection section: '+cid)
    collections={item['id'] for item in data['collections']}
    for entry in data['entries']:
        for key in ['id','title','summary','section','topic','kind','source','url','updated','featured']:
            require(key in entry,f'Entry missing {key}: {entry.get("id")}')
        require(entry['id'] not in ids,'Duplicate entry id: '+entry['id']); ids.add(entry['id'])
        require(entry['source'] not in sources,'Duplicate source: '+entry['source']); sources.add(entry['source'])
        require(entry['topic'] in topics,'Unknown topic: '+entry['id'])
        require(entry.get('section') in sections,'Unknown section: '+entry['id'])
        target=(ROOT/entry['source']).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file(),'Missing/unsafe source: '+entry['source'])
        if entry['source'].endswith('.md'):
            cid=entry.get('collection')
            require(cid in collections,'Missing collection: '+entry['source'])
            require(entry.get('section')==collection_sections.get(cid),'Entry section differs from collection: '+entry['id'])
            require(bool(entry.get('group')),'Missing sidebar group: '+entry['source'])
            rel=entry['source'][len(cid)+1:]
            expected=cid+'/#/'+('' if rel=='README.md' else rel.removesuffix('.md'))
            require(entry['url']==expected,'Incorrect Docsify route: '+entry['source'])
        else: require(entry['url']==entry['source'],'HTML URL must point to source: '+entry['source'])
        require(not entry.get('updated') or bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}',entry['updated'])),'Invalid date: '+entry['id'])
    for cid in collections:
        for path in (ROOT/cid).rglob('*.md'):
            if path.name.startswith('_'): continue
            require(path.relative_to(ROOT).as_posix() in sources,'Unregistered Markdown: '+str(path.relative_to(ROOT)))
    public=[p for p in ROOT.rglob('*.html') if not any(part in ['.git','templates','assets','design'] for part in p.relative_to(ROOT).parts)]
    html_entries={entry['source']:entry for entry in data['entries'] if entry['source'].endswith('.html')}
    parsed={p.resolve():Page(p.read_text(encoding='utf-8')) for p in public}
    for path,page in parsed.items():
        rel=path.relative_to(ROOT).as_posix(); text=path.read_text(encoding='utf-8')
        require(page.body and '<html lang="zh-CN"' in text,'Missing HTML document structure: '+rel)
        require(page.styles==0,'Inline stylesheet: '+rel)
        require(page.inline_scripts==0,'Inline script: '+rel)
        require('assets/css/tokens.css' in text and 'assets/css/site.css' in text,'Missing common design: '+rel)
        require(text.count('<header class="lf-header">')==1,'Expected one shared header: '+rel)
        require(text.count('<footer class="lf-footer">')==1,'Expected one shared footer: '+rel)
        require('user-scalable=no' not in text and 'maximum-scale=1' not in text,'Zoom is disabled: '+rel)
        require(not page.duplicates,'Duplicate element IDs: '+rel+' '+str(page.duplicates))
        if rel not in ['index.html','about.html','topics.html','diagrams.html'] and rel not in [c+'/index.html' for c in collections]:
            require(rel in sources,'Unregistered HTML: '+rel)
            if rel in html_entries:
                require(f'data-page="{html_entries[rel]["id"]}"' in text,'Page id does not match registry: '+rel)
        for ref in page.refs:
            url=urlsplit(ref)
            if url.scheme or url.netloc or ref.startswith('data:'): continue
            relative=unquote(url.path)
            target=(ROOT/relative.lstrip('/') if relative.startswith('/') else path.parent/relative).resolve() if relative else path
            if target.is_dir(): target=target/'index.html'
            require(target.is_relative_to(ROOT.resolve()) and target.exists(),f'Broken local link: {rel} -> {ref}')
            if not target.is_file(): continue
            if url.fragment.startswith('/') and target.name=='index.html':
                route=unquote(url.fragment.split('?')[0]).strip('/') or 'README'
                doc=target.parent/(route if route.endswith('.md') else route+'.md')
                require(doc.is_file(),f'Broken document route: {rel} -> {ref}')
            elif url.fragment and target in parsed:
                require(unquote(url.fragment) in parsed[target].ids,f'Broken anchor: {rel} -> {ref}')
    for path in (ROOT/'assets/css/pages').glob('*.css'):
        text=path.read_text(encoding='utf-8')
        require('@layer legacy' in text and '.lf-legacy[data-legacy=' in text,'Unscoped legacy stylesheet: '+path.name)
    require((ROOT/'AGENTS.md').exists(),'Missing root AGENTS.md')
    require((ROOT/'design/DESIGN_SYSTEM.md').exists(),'Missing design contract')
    node=shutil.which('node')
    require(bool(node),'Node.js is required for JavaScript syntax checks')
    if node:
        js=list((ROOT/'assets/js').rglob('*.js'))+[ROOT/'XCCL/hccl-selector/app.js']
        for path in js:
            result=subprocess.run([node,'--check',str(path)],capture_output=True,text=True,encoding='utf-8')
            require(result.returncode==0,'JavaScript syntax: '+str(path.relative_to(ROOT))+'\n'+result.stderr)
    if ERRORS:
        print('\n'.join(ERRORS)); print(f'FAILED: {len(ERRORS)} issue(s).'); raise SystemExit(1)
    print(f'PASS: {len(public)} pages, {len(sources)} indexed entries, local links, design contracts and JavaScript syntax.')

if __name__=='__main__': main()
