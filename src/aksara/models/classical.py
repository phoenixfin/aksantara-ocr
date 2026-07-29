"""Classical (non-deep) baselines.

A dataset paper needs these. If HOG+SVM lands within a point or two of a
fine-tuned ViT, that is itself the finding — it says the dataset is too easy,
and it is far better to report that yourself than to have a reviewer notice it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm


def extract_features(paths, image_size: int = 64, feature: str = "hog") -> np.ndarray:
    """Vectorize images with a fixed, non-learned representation."""
    vectors = []
    for path in tqdm(paths, desc=f"features:{feature}", leave=False):
        with Image.open(path) as img:
            image = np.asarray(img.convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0

        if feature == "hog":
            vectors.append(
                hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                )
            )
        elif feature == "pixels":
            vectors.append(image.ravel())
        else:
            raise ValueError(f"Unknown feature type: {feature!r}")

    return np.stack(vectors)


# Chosen to actually finish at ~70k samples x 1764 HOG dims x 889 classes on a
# CPU. The original exact-kernel SVC is O(n^2)-O(n^3) in libsvm and would run for
# days; even RF-300 and saga-logistic are hours at this scale. These are the
# leaner, scale-appropriate substitutes:
#   svm_linear     -> LinearSVC (liblinear), linear in n
#   svm_rbf_approx -> Nystroem RBF feature map + LinearSVC (scalable RBF)
#   random_forest  -> 150 trees, depth-capped, fully parallel
# knn has a trivial fit but a heavy predict; it stays because it is the canonical
# non-parametric baseline and its cost is bounded by the test-set size.
CLASSIFIERS = {
    "svm_linear": lambda: make_pipeline(
        StandardScaler(), LinearSVC(C=1, dual="auto", max_iter=2000)
    ),
    "svm_rbf_approx": lambda: make_pipeline(
        StandardScaler(),
        # gamma=None -> Nystroem uses 1/n_features (it rejects the "scale"
        # string that SVC accepts). n_components trades accuracy for speed.
        Nystroem(kernel="rbf", gamma=None, n_components=300, random_state=42),
        LinearSVC(C=10, dual="auto", max_iter=2000),
    ),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=150, max_depth=40, n_jobs=-1, random_state=42
    ),
}

# Default lean set for the run script: one non-parametric, one linear SVM, one
# ensemble. Enough to characterize non-deep performance without an all-day run.
DEFAULT_MODELS = ["knn", "svm_linear", "random_forest"]


def build_classical(name: str, seed: int = 42):
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classical model {name!r}. Available: {sorted(CLASSIFIERS)}")
    model = CLASSIFIERS[name]()
    # Not every estimator exposes random_state; set it where it exists so the
    # classical arm is as reproducible as the deep arm.
    if hasattr(model, "random_state"):
        model.random_state = seed
    return model
