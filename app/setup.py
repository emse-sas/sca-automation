import argparse

import ui

from lib.utils import operation_decorator


@operation_decorator("setup.py", "\nexiting...")
def main():
    ui.create_logs_dir()
    ui.create_images_dir()


argp = argparse.ArgumentParser(
    description="Create directories and data for the application.")

if __name__ == "__main__":
    main()
