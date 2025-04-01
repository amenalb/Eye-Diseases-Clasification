import os
from collections import defaultdict

# zliczanie iloci rodzajów danych rozrzerzeń
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
