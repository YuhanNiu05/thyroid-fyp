"""Download immutable official evidence; never overwrite existing files."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def download(url, target, md5=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as src, tmp.open('wb') as out:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
        tmp.rename(target)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    if md5 and digest != md5:
        raise RuntimeError(f'MD5 mismatch: {target}: {digest} != {md5}')
    print(json.dumps({'file': str(target), 'bytes': target.stat().st_size, 'md5': digest, 'url': url}), flush=True)
    return target

def extract(path, destination):
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite {destination}')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f'Unsafe ZIP member: {name}')
        z.extractall(destination)
    print('Extracted', destination, flush=True)

def main():
    metadata = json.loads((ROOT / 'evidence/figshare_metadata.json').read_text())
    sha = json.loads((ROOT / 'evidence/github_commit.json').read_text())['sha']
    jobs = [(f['download_url'], ROOT / 'downloads' / f['name'], f['computed_md5']) for f in metadata['files']]
    jobs.append((f'https://codeload.github.com/Qingyanyichen/TN5000-2025/zip/{sha}', ROOT / 'downloads' / f'github-{sha}.zip', None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda args: download(*args), jobs))
    for path, subdir in zip(paths, ['data/official', 'vendor/figshare', 'vendor/github']):
        extract(path, ROOT / subdir)

if __name__ == '__main__':
    main()
