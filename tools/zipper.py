import shutil
import sys
import os

def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Zipped '{folder_path}' to '{output_zip}.zip'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <folder_path> <output_zip_name>")
        sys.exit(1)
    folder = sys.argv[1]
    output = sys.argv[2]
    zip_folder(folder, output)