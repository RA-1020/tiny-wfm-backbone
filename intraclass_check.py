import matplotlib.pyplot as plt
from dataset import RadioMLSpectrograms

ds = RadioMLSpectrograms("radioml_subset.npz")
target_class = ds.classes.index("BPSK")

idxs = [i for i in range(len(ds)) if ds.y[i] == target_class][:5]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, i in zip(axes, idxs):
    img, _ = ds[i]
    ax.imshow(img[0], aspect="auto", cmap="viridis")
    ax.axis("off")
plt.suptitle(f"5 examples of class: {ds.classes[target_class]}")
plt.savefig("intraclass_bpsk.png", dpi=120)
plt.show()
