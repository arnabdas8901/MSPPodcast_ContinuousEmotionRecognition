import os
import shutil

train_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/whisper_train_list.txt"
val_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/whisper_val_list.txt"
test_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/whisper_test_list.txt"

target_train_path = "/ds/audio/MSP_Podcast/whisper_medium_features/test/"

with open(test_path) as train_file:
    for line in train_file:
        source_file_path = line.split("|")[0]
        labels = ",".join(line.split("|")[1:])[:-1]
        print(labels)
        source_file_key = source_file_path.split("/")[-1].split(".")[0]
        print(source_file_key)
        shutil.move(source_file_path, os.path.join(target_train_path, source_file_key+".pt"))
        with open(os.path.join(target_train_path, source_file_key+".cls"), "w") as cls_file:
            cls_file.write(labels)
        cls_file.close()
train_file.close()