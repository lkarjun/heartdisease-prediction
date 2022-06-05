from pathlib import Path
import yaml


cwd = Path(__file__).resolve().parent.parent

change_path = cwd/'model/model_tracking/0'


def change(path, artifacts = False):
    with open(path/'meta.yaml') as file:
        data = yaml.load(file, yaml.FullLoader)

    if artifacts:
        data['artifact_uri'] = str(path/'artifacts')
    else:
        data['artifact_location'] = str(path)

    with open(path/'meta.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    print(data, '\n')


def main():
    
    change(change_path)

    for folder in change_path.iterdir():
        if folder.is_dir():
            change(folder, artifacts=True)


if __name__ == "__main__":
    main()