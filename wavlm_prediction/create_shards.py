import os
import shutil
import tarfile

destination_path = "/ds/audio/MSP_Podcast/wavlm_large_features/"
train_path = "/ds/audio/MSP_Podcast/wavlm_large_features/test/"
cls_path = "/ds/audio/MSP_Podcast/whisper_medium_features/test/"

shard_num = 0
idx = 0
for files in os.listdir(train_path):
    if ".pt" in files:
        print(files)
        base_name = files.split(".")[0]
        complete_path = os.path.join(train_path, files)
        cls_file_path = os.path.join(cls_path, base_name+".cls")
        copied_path = os.path.join(train_path, base_name+".cls")
        shutil.copy(cls_file_path, copied_path)
        print("index", idx)
        if idx > 0 and idx%500 == 0:
            shard_num +=1
        tar_file_name = "test-" + str(shard_num).zfill(6) + ".tar"
        print(tar_file_name)

        tar_path = os.path.join(destination_path, tar_file_name)
        if os.path.exists(tar_path):
            tar = tarfile.open(tar_path, "a")
        else:
            tar = tarfile.open(tar_path, "w")
        tar.add(complete_path)
        tar.add(copied_path)
        tar.close()
        idx += 1
