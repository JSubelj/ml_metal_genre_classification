from init_model import create_model
import dataset_tools
import config
import time

if __name__=="__main__":
    start_time = time.time()
    model = create_model()
    model_creation_time = time.time()-start_time

    start_time_ = time.time()
    validate_X, validate_y = dataset_tools.import_dataset("validate")
    dataset_import_time = time.time() - start_time_

    start_time_ = time.time()
    model.load(config.model_directory+config.model_name)
    model_load_time = time.time() - start_time_

    print("Model loaded!")

    start_time_ = time.time()
    validation_accuracy = model.evaluate(validate_X,validate_y)[0]
    model_evaluation_time = time.time() - start_time_

    print("Validation accuracy:",validation_accuracy)

    with open(config.log_file_validating, "a") as f:
        f.write("---------------------------------------------------------------------------------------------------\n")
        f.write("files: validation - " + str(config.validate_percentage * 100) + "%, test - " + str(
            config.test_percentage * 100) + "%, train - " + str(config.train_percentage * 100) + "% " + "\n")
        f.write(str(len(validate_y))+" validation files.\n")
        f.write("validation accuracy: "+str(validation_accuracy)+"\n\n")
        f.write("model_creation_time: " + str(model_creation_time) + "s\n")
        f.write("dataset import time: " + str(dataset_import_time) + "s\n")
        f.write("model load time: " + str(model_load_time) + "s\n")
        f.write("model evaluation time: " + str(model_evaluation_time) + "s\n")
        f.write("summas summarum: " + str(time.time()-start_time) + "s\n")