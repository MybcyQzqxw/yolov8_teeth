import os
import tarfile
import argparse

def extract_tar_in_place(directory):
    tar_files = [f for f in os.listdir(directory) if f.endswith('.tar')]
    extracted_files = []
    extraction_failed = False

    for filename in tar_files:
        tar_path = os.path.join(directory, filename)
        extract_path = os.path.splitext(tar_path)[0]  # 使用文件名作为解压目录
        print(f"解压中：{tar_path}")

        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=directory)  # 解压到当前目录
                extracted_files.append(extract_path)
            print(f"解压完成：{filename}")
        except Exception as e:
            print(f"解压失败：{filename}, 原因：{e}")
            extraction_failed = True

    if extraction_failed:
        print("部分文件解压失败，删除所有解压出的文件...请重新下载压缩包！")
        for extracted in extracted_files:
            if os.path.exists(extracted):
                if os.path.isdir(extracted):
                    # 递归删除目录内容（排除.gitkeep）
                    for root, dirs, files in os.walk(extracted, topdown=False):
                        for name in files:
                            if name != '.gitkeep':
                                os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    try:
                        os.rmdir(extracted)
                    except OSError:
                        pass  # 如果还存在.gitkeep则不删
                else:
                    if os.path.basename(extracted) != '.gitkeep':
                        os.remove(extracted)
    else:
        print("所有文件解压成功，删除所有 .tar 文件...")
        for tar_path in tar_files:
            tar_full_path = os.path.join(directory, tar_path)
            if os.path.exists(tar_full_path):
                os.remove(tar_full_path)

def main():
    parser = argparse.ArgumentParser(description="解压指定目录下的所有 .tar 文件")
    parser.add_argument('--directory', type=str, default='./datasets/dentalai',
                        help="待解压的目录路径，默认为 './datasets/dentalai'")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误：指定的目录不存在：{args.directory}")
        return

    extract_tar_in_place(args.directory)

if __name__ == '__main__':
    main()
