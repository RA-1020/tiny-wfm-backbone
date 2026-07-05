import matplotlib.pyplot as plt
from dataset import RadioMLSpectrograms

ds = RadioMLSpectrograms("radioml_subset.npz")
print(len(ds), "examples,", len(ds.classes), "classes:", ds.classes)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
shown = set()
for img, label in ds:
    if label not in shown:
        ax = axes.flat[len(shown)]
        ax.imshow(img[0], aspect="auto", cmap="viridis")
        ax.set_title(ds.classes[label]); ax.axis("off")
        shown.add(label)
    if len(shown) == len(ds.classes):
        break
plt.tight_layout()
plt.savefig("spectrogram_sanity.png", dpi=120)
plt.show()
