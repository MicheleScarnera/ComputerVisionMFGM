import mfgm_imaterialist as im

dataset = im.import_rawdata(data_type='training',
                  dataset_size=-1,
                  freeloader_mode=True,
                  attempt_downloading_images=False,
                  delete_orphan_entries=False,
                  save_json=True)

print(dataset)