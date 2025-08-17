import os
import zipfile
import argparse
import getpass

try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False

def extract_zip_in_place(directory):
    """解压指定目录下的所有 .zip 文件"""
    zip_files = [f for f in os.listdir(directory) if f.endswith('.zip')]
    if not zip_files:
        print("没有找到需要解压的 .zip 文件")
        return True
    
    # 使用固定密码
    password = "CVPR2024-OralXrays-9"
    print(f"使用密码进行解压...")
    
    extracted_files = []
    extraction_failed = False

    for filename in zip_files:
        zip_path = os.path.join(directory, filename)
        print(f"解压中：{zip_path}")

        try:
            # 优先尝试pyzipper（支持AES加密）
            if HAS_PYZIPPER:
                try:
                    with pyzipper.AESZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.setpassword(password.encode('utf-8'))
                        zip_ref.extractall(directory)
                    print(f"解压完成：{filename}")
                    extracted_files.append(filename)
                    continue
                except Exception as pyzipper_error:
                    print(f"pyzipper解压失败：{filename}, 尝试标准zipfile...")
            
            # 尝试标准zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory, pwd=password.encode('utf-8'))
            print(f"解压完成：{filename}")
            extracted_files.append(filename)
        except Exception as e:
            print(f"解压失败：{filename}, 原因：{e}")
            extraction_failed = True

    if extraction_failed:
        print("部分文件解压失败，请检查压缩包完整性或密码是否正确！")
        return False
    else:
        print("所有文件解压成功，删除所有 .zip 文件...")
        for zip_file in zip_files:
            zip_full_path = os.path.join(directory, zip_file)
            if os.path.exists(zip_full_path):
                os.remove(zip_full_path)
        return True
def main():
    parser = argparse.ArgumentParser(description="解压指定目录下的所有 .zip 文件")
    parser.add_argument('--directory', type=str, default='./datasets/oralxrays9',
                        help="待解压的目录路径，默认为 './datasets/oralxrays9'")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误：指定的目录不存在：{args.directory}")
        return

    success = extract_zip_in_place(args.directory)
    
    if success:
        print("\n解压完成！数据集目录结构：")
        print(f"├── {args.directory}/")
        print("    ├── annotations/")
        print("    │   ├── instances_train2017.json")
        print("    │   └── instances_val2017.json")
        print("    ├── train2017/          # 训练图片")
        print("    └── val2017/            # 验证图片")

if __name__ == '__main__':
    main()
