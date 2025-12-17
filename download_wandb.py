import wandb
import os

api = wandb.Api()

# 下载两个run的所有.pt文件
runs_info = [
    ('nvz7c8na', '/root/gpufree-data/TWIST2/legged_gym/logs/taks_t1_stu_future/nvz7c8na'),
    ('bxea4hiz', '/root/gpufree-data/TWIST2/legged_gym/logs/taks_t1_stu_future/bxea4hiz'),
]

for run_id, save_dir in runs_info:
    os.makedirs(save_dir, exist_ok=True)
    run = api.run(f'xhz2082416211-chinese/taks_t1_stu_future/{run_id}')
    
    print(f'\n=== 正在下载 run {run_id} ===')
    for file in run.files():
        if file.name.endswith('.pt') or file.name == 'config.yaml':
            target_path = os.path.join(save_dir, file.name)
            if os.path.exists(target_path):
                print(f'  跳过(已存在): {file.name}')
                continue
            print(f'  下载: {file.name}')
            file.download(root=save_dir, replace=True)

print('\n=== 全部下载完成 ===')
