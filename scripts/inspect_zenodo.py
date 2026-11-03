import urllib.request
import json
import sys

url = "https://zenodo.org/api/records/3384388"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("Record Title:", data.get("metadata", {}).get("title"))
        print("License:", data.get("metadata", {}).get("license"))
        print("Description excerpt:", data.get("metadata", {}).get("description")[:300])
        print("Files available:")
        for f in data.get("files", []):
            size_mb = f.get("size", 0) / (1024 * 1024)
            print(f"  - {f.get('key')}: {size_mb:.1f} MB (download: {f.get('links', {}).get('self')})")
except Exception as e:
    print(f"Error querying Zenodo API: {e}", file=sys.stderr)
