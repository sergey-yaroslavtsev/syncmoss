"""
Build script for SYNCmoss macOS app bundle.

Usage examples:
    python macos_bundle.py
    python macos_bundle.py --skip --dist-dir bundle/MacOS/arm64
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BUNDLE_DIR)
SYNCMOSS_DIR = os.path.join(ROOT_DIR, "syncmoss")
SPEC_FILE = os.path.join(BUNDLE_DIR, "MacOS.spec")


def abs_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(ROOT_DIR, path_value))


def run_pyinstaller(dist_dir: str, work_dir: str):
    print("=" * 60)
    print("Running PyInstaller (macOS)...")
    print("=" * 60)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        SPEC_FILE,
        "--distpath",
        dist_dir,
        "--workpath",
        work_dir,
        "--noconfirm",
    ]
    subprocess.check_call(cmd, cwd=BUNDLE_DIR)
    print()


def copy_resources(app_dir: str):
    print("=" * 60)
    print(f"Copying resources -> {app_dir}")
    print("=" * 60)

    app_contents_dir = os.path.join(app_dir, "Contents")
    app_macos_dir = os.path.join(app_contents_dir, "MacOS")
    if not os.path.isdir(app_macos_dir):
        raise FileNotFoundError(f"Expected app directory is missing: {app_macos_dir}")

    folder_pairs = [
        (os.path.join(SYNCMOSS_DIR, "icons"), os.path.join(app_macos_dir, "icons")),
        (os.path.join(SYNCMOSS_DIR, "parameters"), os.path.join(app_macos_dir, "parameters")),
        (os.path.join(SYNCMOSS_DIR, "Library"), os.path.join(app_macos_dir, "Library")),
    ]
    for src, dst in folder_pairs:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  OK: {os.path.relpath(dst, app_macos_dir)}/")

    root_files = [
        os.path.join(SYNCMOSS_DIR, "theme_dark.json"),
        os.path.join(SYNCMOSS_DIR, "theme_light.json"),
        os.path.join(SYNCMOSS_DIR, "Calibration.dat"),
        os.path.join(ROOT_DIR, "COPYING.txt"),
        os.path.join(ROOT_DIR, "NOTICE.txt"),
        os.path.join(ROOT_DIR, "LICENSE"),
    ]
    for fpath in root_files:
        if os.path.isfile(fpath):
            shutil.copy2(fpath, app_macos_dir)
            print(f"  OK: {os.path.basename(fpath)}")
        else:
            print(f"  WARNING: not found: {fpath}")

    print()


def find_app_dir(dist_dir: str) -> str:
    """Find SYNCmoss.app in common PyInstaller output layouts."""
    candidates = [
        os.path.join(dist_dir, "SYNCmoss", "SYNCmoss.app"),
        os.path.join(dist_dir, "SYNCmoss.app"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    matches = glob.glob(os.path.join(dist_dir, "**", "SYNCmoss.app"), recursive=True)
    for match in matches:
        if os.path.isdir(match):
            return match

    raise FileNotFoundError(f"Could not find SYNCmoss.app under dist directory: {dist_dir}")


def verify(app_dir: str):
    print("=" * 60)
    print("Verifying ...")
    print("=" * 60)

    required = [
        "Contents/MacOS/SYNCmoss",
        "Contents/MacOS/icons/icon_r.ico",
        "Contents/MacOS/icons/CheckBox.png",
        "Contents/MacOS/parameters/Be.txt",
        "Contents/MacOS/parameters/KB.txt",
        "Contents/MacOS/theme_dark.json",
        "Contents/MacOS/theme_light.json",
    ]
    ok = True
    for rel in required:
        full = os.path.join(app_dir, rel)
        if os.path.exists(full):
            print(f"  OK: {rel}")
        else:
            print(f"  MISSING: {rel}")
            ok = False

    if ok:
        print("\nAll checks passed")
    else:
        print("\nSome files are MISSING - build may be incomplete!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build SYNCmoss macOS bundle")
    parser.add_argument("--skip", action="store_true", help="Skip PyInstaller, only re-copy resources")
    parser.add_argument("--dist-dir", default="bundle/MacOS", help="PyInstaller dist directory")
    parser.add_argument("--work-dir", default="bundle/build/MacOS", help="PyInstaller work directory")
    args = parser.parse_args()

    dist_dir = abs_path(args.dist_dir)
    work_dir = abs_path(args.work_dir)
    if not args.skip:
        run_pyinstaller(dist_dir=dist_dir, work_dir=work_dir)

    app_dir = find_app_dir(dist_dir)

    copy_resources(app_dir=app_dir)
    verify(app_dir=app_dir)


if __name__ == "__main__":
    main()
