from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ROBOTIS/omy_f3m_stack_red_cup_5",
    repo_type="dataset",
    local_dir="/home/cubox/ochansol/isaac_code/python/VLA_data_collect/robotis_dataset_sample/omy_f3m_stack_red_cup_5",
    local_dir_use_symlinks=False
)