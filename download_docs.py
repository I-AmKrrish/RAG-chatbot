import os
import re
import urllib.request
import urllib.error
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEX_URL = "https://code.claude.com/docs/llms.txt"
DOCS_DIR = "claude_docs"

def fetch_index():
    print(f"Fetching index from {INDEX_URL}...")
    req = urllib.request.Request(
        INDEX_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8')
    return content

def extract_urls(index_content):
    # Regex to find links like https://code.claude.com/docs/en/...md
    pattern = r"https://code\.claude\.com/docs/en/[\w\-\./]+\.md"
    urls = re.findall(pattern, index_content)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped

def download_file(url, base_dir=DOCS_DIR, retries=3):
    # Extract the relative path after '/docs/en/'
    prefix = "https://code.claude.com/docs/en/"
    if not url.startswith(prefix):
        print(f"Skipping non-conforming URL: {url}")
        return False
    
    rel_path = url[len(prefix):]
    dest_path = os.path.join(base_dir, *rel_path.split('/'))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(dest_path, "wb") as f:
                    f.write(response.read())
            print(f"Downloaded: {rel_path}")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"404 Not Found for {url}, skipping")
                return False
            print(f"HTTP Error {e.code} downloading {url} (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"Error downloading {url} (attempt {attempt}/{retries}): {e}")
        
        if attempt < retries:
            time.sleep(attempt * 2)
            
    print(f"Failed to download {url} after {retries} attempts.")
    return False

def main():
    try:
        index_content = fetch_index()
    except Exception as e:
        print(f"Failed to fetch index: {e}")
        return

    urls = extract_urls(index_content)
    print(f"Found {len(urls)} documentation URLs to download.")
    
    start_time = time.time()
    success_count = 0
    
    # Download concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"Exception for {url}: {e}")
                
    elapsed = time.time() - start_time
    print(f"\nCompleted download of {success_count}/{len(urls)} files in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
