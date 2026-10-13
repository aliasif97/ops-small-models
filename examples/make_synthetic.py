#!/usr/bin/env python3
"""Write small synthetic datasets so both pipelines can be run end to end without any private data.

  python3 examples/make_synthetic.py
Writes:
  thread-classifier/data/parts/part_1_synthetic.txt   (¦-separated rows: rn¦role¦category¦company¦direction¦sender_domain¦subject¦text)
  price-extractor/data/synthetic/catalog.jsonl        (price records in the catalogue schema)
  price-extractor/data/synthetic/raw/*.txt            (the quotation texts those records came from)
Deterministic (seed 7). Companies, products and prices are invented.
"""
import json, os, random, pathlib
random.seed(7)
ROOT = pathlib.Path(__file__).resolve().parent.parent
ROLES = {
 "customer_rfq": ("i", ["Request for quotation: {p}", "RFQ {p} for {site}", "Price inquiry {p}"], ["Please quote {q} {u} of {p} delivered to {site}.", "Kindly send your best price and lead time for {p}."]),
 "customer_followup": ("i", ["RE: RFQ {p}", "Follow up on quotation {p}"], ["Any update on the quotation for {p}?", "We are still waiting for your offer on {p}."]),
 "our_rfq_to_vendor": ("o", ["NLS-RFQ-2026-{n}: {p}", "Request for quotation {p}"], ["Please find attached our RFQ for {q} {u} of {p}. Reply by {d}.", "We request your offer for {p}, {q} {u}, CIF Dammam."]),
 "vendor_pitch": ("i", ["Introducing {co} {p}", "Partnership opportunity: {p}"], ["We manufacture {p} and would like to introduce our range.", "Our {p} are certified and competitively priced."]),
 "prequal_registration": ("i", ["Vendor registration portal", "Prequalification documents required"], ["Please complete the vendor registration for {co}.", "Submit your prequalification file through the portal."]),
 "admin_noise": ("i", ["Invoice #{n} paid", "Your subscription renewal", "Delivery failed"], ["Automated message. Do not reply.", "This is a system notification."]),
 "other": ("i", ["Meeting next week", "Conference invitation", "Holiday schedule"], ["Let us meet on Tuesday to discuss {p}.", "You are invited to the industry conference."]),
}
PRODUCTS = ["runway edge lights", "12 inch inset base plates", "250 kVA cast resin transformer", "XLPE armoured cable 4x95", "obstruction light L-810", "constant current regulator 10 kW", "PAPI unit", "helipad perimeter lights", "11 kV switchgear panel", "glove set"]
UNITS = ["pcs", "m", "sets", "units"]; SITES = ["Riyadh", "Dammam", "Jeddah", "Tabuk", "Al Khobar"]
COS = ["Delta Lighting Co", "Northern Cable Ltd", "Blue Dune Electrical", "Gulf Aero Supply", "Meridian Lamps GmbH"]
DOMS = ["delta-lighting.example", "northerncable.example", "bluedune.example", "gulfaero.example", "meridianlamps.example"]
# classifier parts file
out = ROOT / "thread-classifier/data/parts"; out.mkdir(parents=True, exist_ok=True)
rows = []; rn = 0
weights = {"customer_rfq": 17, "customer_followup": 5, "our_rfq_to_vendor": 38, "vendor_pitch": 8, "prequal_registration": 2, "admin_noise": 20, "other": 10}
for role, w in weights.items():
    for _ in range(w * 6):
        rn += 1; d, subs, bodies = ROLES[role]; i = random.randrange(5)
        f = dict(p=random.choice(PRODUCTS), q=random.randint(2, 500), u=random.choice(UNITS), site=random.choice(SITES), n=random.randint(100, 999), co=COS[i], d="30 Oct")
        rows.append("¦".join([str(rn), role, "aviation" if "light" in f["p"] or "PAPI" in f["p"] else "non-aviation", COS[i], d, DOMS[i], random.choice(subs).format(**f), random.choice(bodies).format(**f)]))
(out / "part_1_synthetic.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
# extractor catalogue + raw texts
pe = ROOT / "price-extractor/data/synthetic"; (pe / "raw").mkdir(parents=True, exist_ok=True)
cat = []
for k in range(60):
    co = random.randrange(5); ref = f"Q-{1000+k}"; lines = []
    for j in range(random.randint(1, 8)):
        p = random.choice(PRODUCTS); q = random.randint(1, 200); up = round(random.uniform(8, 4200), 2); u = random.choice(UNITS)
        lines.append((p, q, u, up))
    txt = [f"{COS[co]}\nQuotation {ref}\nDate: 2026-0{random.randint(1,9)}-1{random.randint(0,9)}\nValidity: 30 days\nCIF Dammam, USD\n\nItem  Description  Qty  Unit  Unit price  Total"]
    for n, (p, q, u, up) in enumerate(lines, 1):
        txt.append(f"{n}  {p}  {q}  {u}  {up:,.2f}  {q*up:,.2f}")
    txt.append(f"\nTotal USD {sum(q*up for _,q,_,up in lines):,.2f}")
    path = pe / "raw" / f"{ref}.txt"; path.write_text("\n".join(txt), encoding="utf-8")
    for (p, q, u, up) in lines:
        cat.append(dict(origin="email", category="AGL", item=p, spec="", qty=q, unit=u, unit_price=up, ext_price=round(q*up,2), ccy="USD", basis="CIF", supplier=COS[co], doc_type="quote", quote_ref=ref, doc_date="2026-05-10", valid_until="2026-06-09", project="", direction="received", confidence=0.95, source_file=f"{ref}.txt", source_path=str(path), msg_id=ref, notes=""))
(pe / "catalog.jsonl").write_text("\n".join(json.dumps(r) for r in cat) + "\n", encoding="utf-8")
print(f"classifier rows: {len(rows)}  extractor docs: 60  records: {len(cat)}")
