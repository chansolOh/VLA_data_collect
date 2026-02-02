import glob
import os
import imageio.v2 as imageio
import pandas as pd
import json 

global_index = 0

def obs_to_action(action_data, index):
    stride = 2*4 ## stride = 2 , because 25fps , 4 = downsample factor
    action_idx = index + stride
    if action_idx >= len(action_data):
        action_idx = len(action_data) -1
    return align_joint_positions(action_data[action_idx]["robot"]["joint_positions"], action_data[action_idx]["robot"]["joint_names"])

def align_joint_positions(joint_positions, joint_names):
    ordered_joint_positions = []
    for name in joint_names:
        idx = joint_names.index(name)
        if "joint" in name:
            ordered_joint_positions.append(joint_positions[idx])
    return ordered_joint_positions

def save_data_to_parquet(action_path,rgb_path, output_path,ep_num ):
    global global_index
    files = [i.strip(".png") for i in sorted(os.listdir(f"{rgb_path}"))]
    with open( os.path.join(action_path, f"{ep_num:04d}.json"), 'r') as f:
        action_data = json.load(f)
    data = []
    frame_index = 0
    for i, action in enumerate(action_data):
        if f"{action['index']:04d}" not in files: continue

        obs_joint = align_joint_positions(action["robot"]["joint_positions"], action["robot"]["joint_names"])
        data.append({
            "time_stamp": action["time"],
            "frame_index":frame_index,
            "episode_index": ep_num,
            "index" : global_index,
            "task_index":0,
            "observation.state":obs_joint,
            "action":obs_to_action(action_data, i),
        })

        frame_index += 1
        global_index += 1
    df = pd.DataFrame(data)
    df.to_parquet(f"{output_path}/episode_{ep_num:06d}.parquet", engine="pyarrow")



def make_video_from_frames(dir_path, output_video_path):
    if os.path.exists(output_video_path):
        print(f"Video already exists: {output_video_path}, skip...")
        return
    files = sorted(glob.glob(f"{dir_path}/*.png")) 

    writer = imageio.get_writer(
        output_video_path,
        fps=25,                 # 재생 fps
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-preset", "veryfast",   # ultrafast / superfast / veryfast / faster / fast ...
            "-crf", "23",            # 품질(낮을수록 고품질/느림). 18~28 많이 씀
            "-threads", "0",         # ffmpeg가 알아서 코어 최대 사용
        ],
    )

    for f in files:
        writer.append_data(imageio.imread(f))

    writer.close()

    print(f"Saved video to {output_video_path}")



def make_meta(dir_path , ep_num):
    os.makedirs( os.path.join(dir_path, "meta"), exist_ok=True)
    with open( os.path.join(dir_path, "meta", f"episodes_stats.jsonl"), 'w') as f:
        {
            "episode_index": ep_num, 
            "stats": {
                "timestamp": {
                    "min": [0.0], 
                    "max": [38.86666666666667], 
                    "mean": [19.433333333333334], 
                    "std": [11.2294586129849], 
                    "count": [1167]
                    }, 
                "frame_index": {
                    "min": [0], 
                    "max": [1166], 
                    "mean": [583.0], 
                    "std": [336.883758389547], 
                    "count": [1167]
                    }, 
                "episode_index": {
                    "min": [0],
                    "max": [0], 
                    "mean": [0.0], 
                    "std": [0.0], 
                    "count": [1167]
                    }, 
                "index": {
                    "min": [0],
                    "max": [1166], 
                    "mean": [583.0], 
                    "std": [336.883758389547], 
                    "count": [1167]
                    }, 
                "task_index": {
                    "min": [0],
                    "max": [0], 
                    "mean": [0.0], 
                    "std": [0.0], 
                    "count": [1167]
                    }, 
                "observation.images.cam_top": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]], 
                    "max": [[[1.0]], [[1.0]], [[1.0]]], 
                    "mean": [[[0.3916140388711346]], [[0.40037842901769916]], [[0.3763111277189995]]],
                    "std": [[[0.30244417987694927]], [[0.2796817987216563]], [[0.27702456737761777]]], 
                    "count": [199]
                    }, 
                "observation.images.cam_wrist": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[1.0]], [[1.0]], [[1.0]]], 
                    "mean": [[[0.39607380765586475]], [[0.4041940867241622]], [[0.3883846899747226]]], 
                    "std": [[[0.2653885660338933]], [[0.24611077363391534]], [[0.2531036530000994]]], 
                    "count": [199]
                    }, 
                "observation.images.cam_top_depth": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]], 
                    "max": [[[1.0]], [[1.0]], [[1.0]]], 
                    "mean": [[[0.48825800982034356]], [[0.4676861185010018]], [[0.34761237834050424]]], 
                    "std": [[[0.39892488783135327]], [[0.41050925709238145]], [[0.42283181488218013]]], 
                    "count": [199]
                    },
                "observation.images.cam_wrist_depth": {
                    "min": [[[0.0]], [[0.0]], [[0.0]]], 
                    "max": [[[1.0]], [[1.0]], [[1.0]]], 
                    "mean": [[[0.3181467454374268]], [[0.29384132262620294]], [[0.2196194411600486]]], 
                    "std": [[[0.3935528291039326]], [[0.39331582691956063]], [[0.3665136543875934]]], 
                    "count": [199]
                }, 
                "observation.state": {
                    "min": [-0.6701338887214661, -1.5277010202407837, 1.4708718061447144, -1.2256386280059814, 1.4875298738479614, -0.31019967794418335, 0.0], 
                    "max": [0.7446398138999939, 0.33057287335395813, 2.6305134296417236, 0.098702073097229, 2.004948854446411, 1.1392803192138672, 1.0574162006378174], 
                    "mean": [-0.06929301470518112, -0.8340613842010498, 2.2993829250335693, -0.8020712733268738, 1.780045986175537, 0.32429128885269165, 0.2984333336353302], 
                    "std": [0.33759158849716187, 0.7002159953117371, 0.4042917788028717, 0.4326961636543274, 0.16761168837547302, 0.37742751836776733, 0.4078153669834137], 
                    "count": [1167]
                    }, 
                "action": {
                    "min": [-0.6749515533447266, -1.529378890991211, 1.4610631465911865, -1.225650668144226, 1.4533895254135132, -0.31139808893203735, -0.03327898681163788], 
                    "max": [0.7458788752555847, 0.33150172233581543, 2.6305274963378906, 0.11203530430793762, 2.005892276763916, 1.1418392658233643, 1.1848156452178955], 
                    "mean": [-0.06921271979808807, -0.8341448307037354, 2.299360752105713, -0.8016999363899231, 1.7789303064346313, 0.32402724027633667, 0.30626264214515686], 
                    "std": [0.3385952115058899, 0.7007704377174377, 0.4049651622772217, 0.4330361485481262, 0.16823424398899078, 0.3783092200756073, 0.44254228472709656], 
                    "count": [1167]
                    }
                }
            }
        f.write("")

    with open( os.path.join(dir_path, "meta", f"episodes.jsonl"), 'w') as f:
        f.write("")

    with open( os.path.join(dir_path, "meta", f"info.json"), 'w') as f:
        f.write("")

    with open( os.path.join(dir_path, "meta", f"tasks.jsonl"), 'w') as f:
        f.write("")

    

root_path = "/nas/Dataset/VLA/UON/Isaacsim_OMY"
for ep_num in [i for i in os.listdir( os.path.join(root_path, "rgb") ) if os.path.isdir( os.path.join(root_path, "rgb", i) )]:
    for cam_name in ["cam_top", "cam_wrist"]:
        rgb_path = os.path.join(root_path, "rgb", ep_num, cam_name)

        os.makedirs( os.path.join(root_path, "videos", cam_name), exist_ok=True)
        make_video_from_frames(dir_path=rgb_path,
                          output_video_path=os.path.join(root_path, "videos",cam_name, f"{ep_num}.mp4"))
        
    os.makedirs( os.path.join(root_path, "data"), exist_ok=True)
    save_data_to_parquet(action_path=os.path.join(root_path, "action"), 
                         rgb_path=rgb_path, 
                         output_path=os.path.join(root_path, "data"), ep_num=int(ep_num))
    
    make_meta(dir_path=root_path, ep_num=int(ep_num))