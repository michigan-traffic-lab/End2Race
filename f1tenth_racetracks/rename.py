from pathlib import Path

from config import load_racetrack_config


def main():
    module = Path(__file__).resolve().parent
    config = load_racetrack_config().rename

    for map_directory in (path for path in module.iterdir() if path.is_dir()):
        map_name = map_directory.name
        for path in map_directory.iterdir():
            if path.suffix in config.remove_extensions:
                path.unlink()
            elif path.name == f'{map_name}{config.centerline_source_suffix}':
                path.rename(map_directory / f'{map_name}{config.centerline_target_suffix}')
            elif path.name == f'{map_name}{config.raceline_source_suffix}':
                path.rename(map_directory / f'{map_name}{config.raceline_target_suffix}')
            elif path.suffix == '.txt':
                path.rename(map_directory / f'{map_name}{config.donkey_waypoint_suffix}')


if __name__ == '__main__':
    main()
