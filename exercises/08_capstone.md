# Capstone — Embeddings for your own area

Apply the toolkit to a place you know. Choose an area in Greece (or anywhere in Europe/US for TESSERA, anywhere on land for AlphaEarth) of roughly 10 × 10 km that does **not** cross a UTM zone boundary (in Greece: 18°, 24°, 30° E).

## Deliverables

1. **A few-shot land-cover map** (Hands-on 4 recipe) for 2024 with AlphaEarth or TESSERA:
   - 4–6 classes of your choice, 10–30 labels per class from your own knowledge or an existing map,
   - a spatial train/test split and a confusion matrix,
   - the map as a GeoTIFF and a figure.
2. **A change map** (Hands-on 5 recipe) between two years of your choice:
   - the distance histogram with your threshold marked,
   - polygons of changed areas ≥ 1 ha with their size,
   - one sentence per major polygon saying what you think happened (check imagery).
3. **Half a page of interpretation**: what worked, what did not, which product you used and why, what you would do next.

## Submission

A single notebook or PDF plus the two GeoTIFFs. Name files `capstone_<yourname>_<area>.*`.

## Assessment (self-check)

- [ ] Study area and years stated, with a map
- [ ] Labels: source, count per class, how you split train/test spatially
- [ ] Accuracy reported per class, not just overall
- [ ] Threshold justified from the distribution, not copied
- [ ] Product version and licence attribution included
- [ ] Interpretation distinguishes "the embedding changed" from "I believe X happened because…"
