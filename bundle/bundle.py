"""
Build script for SYNCmoss Windows executables.

Usage:
    python bundle.py          # full clean build
    python bundle.py --skip   # skip PyInstaller, only re-copy resources

This script:
  1. Runs PyInstaller with Windows.spec
  2. Copies application resources (icons, parameters, themes, etc.)
     next to the .exe so they are NOT buried inside _internal/
"""

import argparse
import os
import shutil
import subprocess
import sys

# ── Paths ───────────────────────────────────────────────────────
BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BUNDLE_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
SPEC_FILE = os.path.join(BUNDLE_DIR, "Windows.spec")
DIST_DIR = os.path.join(BUNDLE_DIR, "Windows")
BUILD_DIR = os.path.join(BUNDLE_DIR, "build", "Windows")
OUTPUT_DIR = os.path.join(DIST_DIR, "SYNCmoss")  # COLLECT name


def run_pyinstaller():
    """Run PyInstaller using the .spec file."""
    print("=" * 60)
    print("Running PyInstaller …")
    print("=" * 60)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        SPEC_FILE,
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--noconfirm",
    ]
    subprocess.check_call(cmd, cwd=BUNDLE_DIR)
    print()


def copy_resources():
    """Copy application resource files next to the .exe."""
    print("=" * 60)
    print("Copying resources → ", OUTPUT_DIR)
    print("=" * 60)

    # ── Folders to copy verbatim ────────────────────────────────
    folder_pairs = [
        (os.path.join(SRC_DIR, "icons"),      os.path.join(OUTPUT_DIR, "icons")),
        (os.path.join(SRC_DIR, "parameters"), os.path.join(OUTPUT_DIR, "parameters")),
    ]
    for src, dst in folder_pairs:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  ✓ {os.path.relpath(dst, OUTPUT_DIR)}/")

    # ── Individual files to copy to OUTPUT_DIR root ─────────────
    root_files = [
        # from src/
        os.path.join(SRC_DIR, "theme_dark.json"),
        os.path.join(SRC_DIR, "theme_light.json"),
        os.path.join(SRC_DIR, "Calibration.dat"),
        # from repo root
        os.path.join(ROOT_DIR, "COPYING.txt"),
        os.path.join(ROOT_DIR, "NOTICE.txt"),
        os.path.join(ROOT_DIR, "LICENSE"),
    ]
    for f in root_files:
        if os.path.isfile(f):
            shutil.copy2(f, OUTPUT_DIR)
            print(f"  ✓ {os.path.basename(f)}")
        else:
            print(f"  ⚠ not found: {f}")

    # ── Remove stale copies from _internal/ (if any) ───────────
    internal = os.path.join(OUTPUT_DIR, "_internal")
    for name in ("icons", "parameters"):
        stale = os.path.join(internal, name)
        if os.path.isdir(stale):
            shutil.rmtree(stale)
            print(f"  🗑 removed _internal/{name}/")
    for name in ("theme_dark.json", "theme_light.json", "Calibration.dat",
                 "COPYING.txt", "NOTICE.txt", "LICENSE"):
        stale = os.path.join(internal, name)
        if os.path.isfile(stale):
            os.remove(stale)
            print(f"  🗑 removed _internal/{name}")

    print()


def verify():
    """Quick sanity check that critical files exist."""
    print("=" * 60)
    print("Verifying …")
    print("=" * 60)
    required = [
        "SYNCmoss.exe",
        "SYNCmoss_console.exe",
        "icons/icon_r.ico",
        "icons/CheckBox.png",
        "icons/CheckBox_L.png",
        "icons/CheckBox_L2.png",
        "icons/UD.png",
        "icons/DU.png",
        "parameters/Be.txt",
        "parameters/KB.txt",
        "parameters/GCMS.txt",
        "parameters/INSexp.txt",
        "parameters/INSint.txt",
        "theme_dark.json",
        "theme_light.json",
        "Calibration.dat",
        "COPYING.txt",
        "NOTICE.txt",
        "LICENSE",
    ]
    ok = True
    for rel in required:
        full = os.path.join(OUTPUT_DIR, rel)
        if os.path.exists(full):
            print(f"  ✓ {rel}")
        else:
            print(f"  ✗ MISSING: {rel}")
            ok = False

    if ok:
        print("\nAll checks passed ✓")
    else:
        print("\nSome files are MISSING — build may be incomplete!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build SYNCmoss Windows bundle")
    parser.add_argument("--skip", action="store_true",
                        help="Skip PyInstaller, only re-copy resources")
    args = parser.parse_args()

    if not args.skip:
        run_pyinstaller()

    copy_resources()
    verify()


if __name__ == "__main__":
    main()
