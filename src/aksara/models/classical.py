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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


CLASSIFIERS = {
    "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    "svm_linear": lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, n_jobs=-1)),
}


def build_classical(name: str, seed: int = 42):
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classical model {name!r}. Available: {sorted(CLASSIFIERS)}")
    model = CLASSIFIERS[name]()
    # Not every estimator exposes random_state; set it where it exists so the
    # classical arm is as reproducible as the deep arm.
    if hasattr(model, "random_state"):
        model.random_state = seed
    return model
