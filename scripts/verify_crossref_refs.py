import urllib.request, urllib.parse, json, os

queries = [
    ("Borobudur_ISPRS", "Borobudur relief 3D deep learning ISPRS"),
    ("Heritage_Seg_ISPRS", "stone relief segmentation deep learning cultural heritage ISPRS 2024"),
    ("CIPA_Waldhausl", "Waldhausl Ogleby 3 by 3 rules CIPA 1994"),
    ("CIPA_2013", "3x3 rules simple photogrammetric documentation cultural heritage CIPA"),
    ("Borobudur_Conservation", "Borobudur temple conservation weathering lichen andesite"),
    ("Guo_2026_HERI3D", "HERI3D Comparative Analysis Traditional Deep Learning UAV Cultural Heritage"),
    ("Lecca_2023", "A Basic Tool for Improving Bad Illuminated Archaeological Pictures Lecca"),
]

out_lines = []
for tag, q in queries:
    url = "https://api.crossref.org/works?rows=4&query=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:research@unika.ac.id"})
    out_lines.append(f"=== {tag}: {q} ===")
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode())
        for it in data["message"]["items"]:
            title = (it.get("title") or [""])[0]
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])[:5]]
            venue = (it.get("container-title") or [""])[0]
            yr = (it.get("published", {}).get("date-parts") or [[None]])[0][0]
            vol = it.get("volume", "")
            iss = it.get("issue", "")
            pg = it.get("page", "")
            doi = it.get("DOI", "")
            out_lines.append(f"  [{yr}] {', '.join(authors)} | \"{title}\" | {venue} | vol={vol} no={iss} pp={pg} | DOI:{doi}")
    except Exception as e:
        out_lines.append(f"  ERR: {e}")

out_path = r"D:\~Ideas n Innovation\25-8-26 --Borobudur\Revision\experiments\verified_crossref_candidates.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print("\n".join(out_lines))
