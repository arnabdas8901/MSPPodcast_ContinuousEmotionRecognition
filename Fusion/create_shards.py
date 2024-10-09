import os
import shutil
import tarfile

destination_path = "/ds/audio/MSP_Podcast/whisper_wavlm_w2v/"
whisper_path = "/ds/audio/MSP_Podcast/whisper_medium_features/test/"
wavlm_path = "/ds/audio/MSP_Podcast/wavlm_large_features/test/"
w2v_path = "/ds/audio/MSP_Podcast/w2vec_bert_features/test/"
split = "test"

shard_num = 0
idx = 0
for files in os.listdir(whisper_path):
    if ".pt" in files:
        print(files)
        base_name = files.split(".")[0]
        whisper_complete_path = os.path.join(whisper_path, files)
        whisper_cls_file_path = os.path.join(whisper_path, base_name+".cls")
        wavlm_complete_path = os.path.join(wavlm_path, files)
        w2v_complete_path = os.path.join(w2v_path, files)
        whisper_target_path = os.path.join(destination_path, split, base_name+".whisper.pt")
        wavlm_target_path = os.path.join(destination_path, split, base_name + ".wavlm.pt")
        w2v_target_path = os.path.join(destination_path, split, base_name + ".w2v.pt")
        cls_file_target_path = os.path.join(destination_path, split, base_name+".cls")

        shutil.copy(whisper_complete_path, whisper_target_path)
        shutil.copy(wavlm_complete_path, wavlm_target_path)
        shutil.copy(w2v_complete_path, w2v_target_path)
        shutil.copy(whisper_cls_file_path, cls_file_target_path)

        print("index", idx)
        if idx > 0 and idx%128 == 0:
            shard_num +=1
        tar_file_name = split+"-" + str(shard_num).zfill(6) + ".tar"
        print(tar_file_name)

        tar_path = os.path.join(destination_path, tar_file_name)
        if os.path.exists(tar_path):
            tar = tarfile.open(tar_path, "a")
        else:
            tar = tarfile.open(tar_path, "w")
        tar.add(whisper_target_path)
        tar.add(wavlm_target_path)
        tar.add(w2v_target_path)
        tar.add(cls_file_target_path)
        tar.close()
        idx += 1
