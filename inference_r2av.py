import os
import sys
import logging
import torch
import json
import re
from tqdm import tqdm
from omegaconf import OmegaConf
from dreamid_omni.utils.io_utils import save_video
from dreamid_omni.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from dreamid_omni.utils.utils import get_arguments
from dreamid_omni.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from dreamid_omni.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from dreamid_omni.dreamid_omni_engine import DreamIDOmniEngine



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)





def load_mixed_data(config):

    data_root = config.get("data_root", "./test_case_2") 
    all_eval_data = []

    modes = {
        'oneip': {'img_folder': 'imgs', 'audio_folder': 'audios', 'json_folder': 'captions'}, 
        'twoip': {'img_folder': 'imgs', 'audio_folder': 'audios', 'json_folder': 'captions'} 
    }
    for mode, subfolders in modes.items():
       
        mode_path = os.path.join(data_root, mode)
        if not os.path.exists(mode_path):
            continue
        img_root = os.path.join(mode_path, subfolders['img_folder'])
        audio_root = os.path.join(mode_path, subfolders['audio_folder'])
        json_root = os.path.join(mode_path, subfolders['json_folder'])
        if not os.path.exists(img_root): continue
        name_list = os.listdir(img_root)
        logging.info(f"Found {len(name_list)} potential test cases in {mode} - {img_root}.")
        for test_name in name_list:
            text_prompt_path = os.path.join(json_root, f'{test_name}.json')
      
            if mode == 'oneip':
        
                image0_path = os.path.join(img_root, test_name, f'{test_name}.png')
                image1_path = None 
                audio0_path = os.path.join(audio_root, test_name, f'{test_name}.wav')
                audio1_path = None 
                check_paths = [text_prompt_path, image0_path, audio0_path]
            else:
          
                image0_path = os.path.join(img_root, test_name, '0.png')
                image1_path = os.path.join(img_root, test_name, '1.png')
                audio0_path = os.path.join(audio_root, test_name, '0.wav')
                audio1_path = os.path.join(audio_root, test_name, '1.wav')
                check_paths = [text_prompt_path, image0_path, image1_path, audio0_path, audio1_path]
            if not all(os.path.exists(p) for p in check_paths):
                logging.warning(f"Skipping '{mode}/{test_name}': missing files.")
                continue
            try:
                with open(text_prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = json.load(f)
           
                text_prompt = re.sub(r'\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]', '', text_prompt, flags=re.DOTALL).strip()
                text_prompt = re.sub(r'\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]', '', text_prompt, flags=re.DOTALL).strip()
                text_prompt = re.sub(r'\[[A-Z_]+\]', '', text_prompt)
                text_prompt = re.sub(r'\n\s*\n', '\n', text_prompt).strip()
            except Exception as e:
                logging.warning(f"Skipping '{test_name}': json error. {e}")
                continue
         
            unique_test_name = f"{mode}_{test_name}"
            all_eval_data.append((unique_test_name, text_prompt, image0_path, image1_path, audio0_path, audio1_path))
    
    return all_eval_data


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # all_eval_data = load_two_person_data(config)
    all_eval_data = load_mixed_data(config)

    logging.info("Loading DreamID Omni Engine...")
    dreamid_omni_engine = DreamIDOmniEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("DreamID Omni Engine loaded!")
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    sp_rank = nccl_info.rank_within_group

    # Load CSV data
 

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:

        
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
        print('use sp_size: ', sp_size)
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size
        print('not use sp_size: ', sp_size)

    # Data distribution - by SP groups
    total_files = len(all_eval_data)

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    for _, (test_name, text_prompt, image0_path, image1_path, audio0_path, audio1_path) in tqdm(enumerate(this_rank_eval_data)):

        video_frame_height_width = config.get("video_frame_height_width", None)
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_cfg_scale = config.get("video_cfg_scale", 3.0)
        video_ref_cfg_scale = config.get("video_ref_cfg_scale", 1.5)
        audio_cfg_scale = config.get("audio_cfg_scale", 4.0)
        audio_ref_cfg_scale = config.get("audio_ref_cfg_scale", 2.0)
        
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        for idx in range(config.get("each_example_n_times", 1)):
            output_path = os.path.join(output_dir, f"{test_name}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}.mp4")
            if os.path.exists(output_path):
                continue
 
            generated_video, generated_audio, generated_image = dreamid_omni_engine.generate(
                                                                    text_prompt=text_prompt,
                                                                    image0_path=image0_path, 
                                                                    image1_path=image1_path,
                                                                    audio0_path=audio0_path,
                                                                    audio1_path=audio1_path, 
                                                                    video_frame_height_width=video_frame_height_width,
                                                                    seed=seed+idx,
                                                                    solver_name=solver_name,
                                                                    sample_steps=sample_steps,
                                                                    shift=shift,
                                                                    video_cfg_scale = video_cfg_scale,
                                                                    video_ref_cfg_scale = video_ref_cfg_scale,
                                                                    audio_cfg_scale = audio_cfg_scale,
                                                                    audio_ref_cfg_scale = audio_ref_cfg_scale,
                                                                    video_negative_prompt=video_negative_prompt,
                                                                    audio_negative_prompt=audio_negative_prompt)
            
            if sp_rank == 0:

                output_path = os.path.join(output_dir, f"{test_name}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}.mp4")
                save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                if generated_image is not None:
                    generated_image.save(output_path.replace('.mp4', '.png'))
        


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)
