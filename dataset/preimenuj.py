import os

dataset_root = 'dataset'

for split in ['train', 'val', 'test']:
    for class_name in os.listdir(f'{dataset_root}/{split}'):
        class_path = f'{dataset_root}/{split}/{class_name}'
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            new_name = filename.replace(' ', '_')
            if new_name != filename:
                os.rename(
                    f'{class_path}/{filename}',
                    f'{class_path}/{new_name}'
                )
                print(f'Preimenovano: {filename} → {new_name}')

print('Končano!')