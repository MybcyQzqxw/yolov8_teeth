import os
import tarfile
import zipfile
import argparse
import shutil


def extract_and_flatten(directory: str):
    """
    解压 dentalx 数据集并扁平化目录结构
    
    处理逻辑：
    1. 解压 training_data.zip 和 validation_data.zip
    2. 把嵌套的目录结构扁平化到 datasets/dentalx/ 根目录
    3. 清理解压的临时目录
    """
    print("开始解压 dentalx 数据集...")
    
    # 解压所有 zip 文件
    zip_files = [f for f in os.listdir(directory) if f.endswith('.zip')]
    
    for zip_file in zip_files:
        zip_path = os.path.join(directory, zip_file)
        extract_path = os.path.splitext(zip_path)[0]
        
        print(f"解压 {zip_file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=extract_path)
            print(f"解压完成: {zip_file}")
        except Exception as e:
            print(f"解压失败: {zip_file}, 错误: {e}")
            continue
    
    # 扁平化 training_data 结构
    training_nested = os.path.join(directory, "training_data", "training_data")
    if os.path.exists(training_nested):
        print("扁平化 training_data 目录结构...")
        for item in os.listdir(training_nested):
            src = os.path.join(training_nested, item)
            dst = os.path.join(directory, item)
            if os.path.exists(dst):
                # 保护.gitkeep文件：先备份，删除目录，再还原
                gitkeep_path = os.path.join(dst, '.gitkeep')
                gitkeep_backup = None
                if os.path.exists(gitkeep_path):
                    gitkeep_backup = os.path.join(directory, f'.gitkeep_backup_{item}')
                    shutil.copy2(gitkeep_path, gitkeep_backup)
                
                shutil.rmtree(dst)
                
                # 如果有.gitkeep备份，在新目录中还原
                if gitkeep_backup and os.path.exists(gitkeep_backup):
                    shutil.move(src, dst)
                    new_gitkeep = os.path.join(dst, '.gitkeep')
                    shutil.copy2(gitkeep_backup, new_gitkeep)
                    os.remove(gitkeep_backup)
                else:
                    shutil.move(src, dst)
            else:
                shutil.move(src, dst)
        # 清理空的嵌套目录
        shutil.rmtree(os.path.join(directory, "training_data"))
    
    # 扁平化 validation_data 结构
    validation_nested = os.path.join(directory, "validation_data", "validation_data") 
    if os.path.exists(validation_nested):
        print("扁平化 validation_data 目录结构...")
        for item in os.listdir(validation_nested):
            src = os.path.join(validation_nested, item)
            dst = os.path.join(directory, f"val_{item}")  # 加 val_ 前缀避免冲突
            if os.path.exists(dst):
                # 保护.gitkeep文件：先备份，删除目录，再还原
                gitkeep_path = os.path.join(dst, '.gitkeep')
                gitkeep_backup = None
                if os.path.exists(gitkeep_path):
                    gitkeep_backup = os.path.join(directory, f'.gitkeep_backup_val_{item}')
                    shutil.copy2(gitkeep_path, gitkeep_backup)
                
                shutil.rmtree(dst)
                
                # 如果有.gitkeep备份，在新目录中还原
                if gitkeep_backup and os.path.exists(gitkeep_backup):
                    shutil.move(src, dst)
                    new_gitkeep = os.path.join(dst, '.gitkeep')
                    shutil.copy2(gitkeep_backup, new_gitkeep)
                    os.remove(gitkeep_backup)
                else:
                    shutil.move(src, dst)
            else:
                shutil.move(src, dst)
        # 清理空的嵌套目录
        shutil.rmtree(os.path.join(directory, "validation_data"))
    
    # 删除原始 zip 文件
    for zip_file in zip_files:
        zip_path = os.path.join(directory, zip_file)
        try:
            os.remove(zip_path)
            print(f"删除原始文件: {zip_file}")
        except Exception:
            pass
    
    print("dentalx 数据集解压和扁平化完成！")
    print("目录结构:")
    for item in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, item)):
            print(f"  📁 {item}/")


def main():
    parser = argparse.ArgumentParser(description="解压并扁平化 dentalx 数据集")
    parser.add_argument('--directory', type=str, default='./datasets/dentalx',
                        help="数据集目录，默认 ./datasets/dentalx")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误：目录不存在: {args.directory}")
        return

    extract_and_flatten(args.directory)


if __name__ == '__main__':
    main()
