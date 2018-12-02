from init_model import create_model
import dataset_tools
import random
import string
import config
import time


if __name__ == "__main__":
    start_time = time.time()
    model = create_model()
    model_create_time = time.time()-start_time
    print("model creation time:", model_create_time)

    start_time_ = time.time()
    train_X, train_y, test_X, test_y = dataset_tools.import_dataset("train")
    dataset_import_time = time.time()-start_time_
    print("import time:", dataset_import_time)


    run_id = "metal genres - " + str(config.batch_size) + " " + ''.join(
        random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

    start_time_ = time.time()
    model.fit(train_X, train_y, n_epoch=config.no_of_epochs, batch_size=config.batch_size, shuffle=True,
              validation_set=(test_X, test_y), snapshot_step=100, show_metric=True, run_id=run_id)
    model_training_time = time.time()-start_time_
    print("training model time:", model_training_time)
    print("model trained")
    start_time_ = time.time()
    model.save(config.model_directory+config.model_name)
    model_saving_time = time.time()-start_time_
    print("saving model time:", model_saving_time)
    summas_summarum_time = time.time()-start_time
    print("summas summarum:", summas_summarum_time)

    with open(config.log_file_training, "a") as f:
        f.write("---------------------------------------------------------------------------------------------------\n")
        f.write("files: validation - "+str(config.validate_percentage*100)+"%, test - "+str(config.test_percentage*100)+"%, train - "+str(config.train_percentage*100)+"% "+"\n")
        f.write("model_creation_time: "+str(model_create_time)+"s\n")
        f.write("dataset import time: "+str(dataset_import_time)+"s\n")
        f.write("model training time: "+str(model_training_time)+"s\n")
        f.write("model saving time: "+str(model_saving_time)+"s\n")
        f.write("summas summarum: "+str(summas_summarum_time)+"s\n")
