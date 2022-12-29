import os
import tensorflow as tf
from dvc_workshop.pipeline.train.io import csv_to_image_data_gen, save_model
from dvc_workshop.pipeline.evaluate.io import load_model, save_json
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, MODEL_NAME
from dvc_workshop.pipeline.evaluate.constants import  SAVE_RESULTS,RESULTS_FILE
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY



def evaluate_model(model_save_path : str, csv_test_path: str,result_save_path : str,results_file:str,image_path:str,target: str) :
    model = load_model(model_save_path)
    test = csv_to_image_data_gen(csv_test_path, image_path, target)
    results_dict = model.evaluate(test, verbose=0, return_dict=True)
    save_json(results_dict,result_save_path,results_file)




def main():
    model_path = os.path.join(SAVE_MODEL,MODEL_NAME)
    evaluate_model(model_save_path=model_path,
    csv_test_path=os.path.join(PREPROCESS_DIRECTORY, "test.csv"),
    result_save_path = SAVE_RESULTS,
    results_file=RESULTS_FILE,
    image_path="Paths",
    target="Labels"
)


if __name__ == "__main__":
    main()
