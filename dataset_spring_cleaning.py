import os
from PIL import Image
from collections import defaultdict

def count_image_color_modes(directory):
    color_mode_count = defaultdict(int)

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    color_mode = img.mode
                    color_mode_count[color_mode] += 1
            except Exception as e:
                continue

    return color_mode_count

def count_image_dimensions(directory):
    dimensions_count = defaultdict(int)

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    dimensions_count[(width, height)] += 1
            except Exception as e:
                continue

    return dimensions_count

def count_file_extensions(directory):
    extension_count = defaultdict(int)
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                extension_count[ext.lower()] += 1

    return extension_count

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = os.path.join(script_dir, "dataset")

    if not os.path.isdir(folder_path):
        print("Podana ścieżka nie jest katalogiem.")
    else:
        extensions = count_file_extensions(folder_path)

        print("\nLiczba plików dla każdego rozszerzenia:")
        for ext, count in sorted(extensions.items()):
            print(f"{ext}: {count}")


        dimensions = count_image_dimensions(folder_path)

        print("\nLiczba plików dla każdego wymiaru:")
        for (width, height), count in sorted(dimensions.items()):
            print(f"Wymiary {width}x{height}: {count}")

        color_modes = count_image_color_modes(folder_path)

        print("\nLiczba plików dla każdego trybu kolorów:")
        for mode, count in sorted(color_modes.items()):
            print(f"Tryb {mode}: {count}")