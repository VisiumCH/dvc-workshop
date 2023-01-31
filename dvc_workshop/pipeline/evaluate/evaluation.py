import os

from image_classification_autotrain.multilabel_classifier import evaluate_model

from dvc_workshop.pipeline.evaluate.constants import RESULTS_FILE, SAVE_RESULTS
from dvc_workshop.pipeline.evaluate.io import load_model, save_json
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import MODEL_NAME, SAVE_MODEL


def main():
    model_path = os.path.join(SAVE_MODEL, MODEL_NAME)

    # load model
    model = load_model(model_path)

    results_dict = evaluate_model(
        model=model,
        csv_test_path=os.path.join(PREPROCESS_DIRECTORY, "test.csv"),
        image_path="Paths",
        target="Labels",
    )

    # save results
    save_json(results_dict, SAVE_RESULTS, RESULTS_FILE)


if __name__ == "__main__":
    main()
