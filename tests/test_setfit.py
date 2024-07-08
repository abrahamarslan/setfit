from src.setfit import __version__, SetFitClassifier
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np

def test_version():
    """
    Test to ensure that the package version is correct.
    """
    assert __version__ == "0.1.4"

def test_e2e():
    """
    End-to-end test to check the full training and prediction cycle.
    """
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]

    # Initialize the classifier with a specific sentence-transformers model
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    # Fine-tune embeddings and train logistic regression head
    clf.fit(docs, labels)

    # Make predictions
    preds = clf.predict(["affirmative", "negative"])
    assert preds.shape == (2,)

    # Predict probabilities
    pproba = clf.predict_proba(["affirmative", "negative"])
    assert pproba.shape == (2, 2)

def test_notfitted_error(tmp_path):
    """
    Test to ensure that the appropriate errors are raised when methods are called on an unfitted classifier.
    """
    docs = ["yay", "boo", "yes", "no", "yeah"]
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")

    # Ensure NotFittedError is raised when calling predict on an unfitted classifier
    with pytest.raises(NotFittedError):
        clf.predict(docs)

    # Ensure NotFittedError is raised when calling predict_proba on an unfitted classifier
    with pytest.raises(NotFittedError):
        clf.predict_proba(docs)

    # Ensure NotFittedError is raised when calling save on an unfitted classifier
    with pytest.raises(NotFittedError):
        clf.save(tmp_path)

def test_save_load(tmp_path):
    """
    Test to check saving and loading of the model.
    """
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]

    # Initialize and fit the classifier
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    clf.fit(docs, labels)

    # Predict with the fitted model
    p1 = clf.predict(docs)

    # Save the model
    clf.save(tmp_path)

    # Load the model and predict again
    clf2 = SetFitClassifier.load(tmp_path)
    p2 = clf2.predict(docs)

    # Ensure predictions from both models are the same
    assert np.array_equal(p1, p2)

def test_get_params():
    """
    Test to ensure get_params method works correctly (required for GridSearchCV).
    """
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    assert clf.get_params()

def test_single_example_no_loop():
    """
    Test to ensure single example training without looping over multiple epochs works correctly.
    """
    docs = ["yes", "no"]
    labels = [1, 0]

    # Initialize and fit the classifier
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    clf.fit(docs, labels)

    # Make predictions
    preds = clf.predict(["affirmative", "negative"])
    assert preds.shape == (2,)

    # Predict probabilities
    pproba = clf.predict_proba(["affirmative", "negative"])
    assert pproba.shape == (2, 2)

def test_multiclass():
    """
    Test to ensure that multiclass classification works correctly.
    """
    docs = ["yay", "boo", "yes", "no", "yeah", "maybe", "don't know"]
    labels = [1, 0, 1, 0, 1, 2, 2]

    n_labels = len(set(labels))

    # Initialize and fit the classifier
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    clf.fit(docs, labels)

    # Predict on new examples
    pred_examples = ["affirmative", "negative", "possibly", "uncertain"]
    n_preds = len(pred_examples)
    preds = clf.predict(pred_examples)
    assert preds.shape == (n_preds,)

    # Predict probabilities
    pproba = clf.predict_proba(pred_examples)
    assert pproba.shape == (n_preds, n_labels)
