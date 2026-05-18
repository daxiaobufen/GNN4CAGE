"""
Script purpose:
- Extract valid structures from CSD and export CIF/XYZ files.
- Build a refcode CSV list for downstream workflow steps.
"""

import os
import json
import tqdm
from ccdc import io


def load_runtime_config():
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_runtime_config()

# ================= Runtime switches =================
# Refer to config setting: extract.test_mode
TEST_MODE = bool(CFG.get("extract", {}).get("test_mode", False))

# 2. Base path (Read from config)
BASE_DIR = CFG.get("paths", {}).get("data_root", "<DATA_ROOT>")

# 3. Output path configuration
if TEST_MODE:
    CIF_OUT_DIR = os.path.join(BASE_DIR, "CSD_CIF_TEST")
    XYZ_OUT_DIR = os.path.join(BASE_DIR, "CSD_Molecule_TEST")
    CSV_OUT_PATH = os.path.join(BASE_DIR, "valid_refcodes_TEST.csv")
    TARGET_COUNT = int(CFG.get("extract", {}).get("target_count", 20))  # Refer to config setting
else:
    CIF_OUT_DIR = os.path.join(BASE_DIR, "CSD_CIF")
    XYZ_OUT_DIR = os.path.join(BASE_DIR, "CSD_Molecule")
    CSV_OUT_PATH = os.path.join(BASE_DIR, "valid_refcodes.csv")

# 4. [Update] Excluded elements list (All metals shielded)
EXCLUDE_ELEMENTS = {
    # --- Group 1: Alkali metals ---
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',

    # --- Group 2: Alkaline earth metals ---
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',

    # --- Groups 3-12: Transition Metals ---
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',

    # --- Lanthanides ---
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',

    # --- Actinides ---
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',

    # --- Post-transition Metals ---
    # If you want to keep Aluminum (Al) or Tin (Sn), etc., please comment out the corresponding element
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'
}
# ===========================================


def batch_process():
    print(f">>> Starting processing program (Mode: {'TEST' if TEST_MODE else 'PRODUCTION'})...")
    print(f"    List output path: {CSV_OUT_PATH}")
    print(f"    Number of shielded metal elements: {len(EXCLUDE_ELEMENTS)}")

    # 1. Create directories
    os.makedirs(CIF_OUT_DIR, exist_ok=True)
    os.makedirs(XYZ_OUT_DIR, exist_ok=True)

    # 2. Connect to CSD
    try:
        csd_reader = io.EntryReader('CSD')
        total_entries = len(csd_reader)
        print(f"Successfully connected to CSD, total database entries: {total_entries}")
    except Exception as e:
        print(f"!!! Failed to connect to CSD: {e}")
        return

    saved_count = 0
    skipped_metal = 0
    skipped_error = 0

    # 3. Prepare iterator
    iterator = range(total_entries)
    if not TEST_MODE:
        iterator = tqdm.tqdm(iterator, desc="Processing CSD", unit="mol")

    # 4. Open CSV file ready for writing
    with open(CSV_OUT_PATH, 'w', encoding='utf-8') as f_csv:

        for i in iterator:
            # Test mode check
            if TEST_MODE and saved_count >= TARGET_COUNT:
                print(f"\n[Test ended] Target extraction count reached: {TARGET_COUNT}")
                break

            try:
                entry = csd_reader[i]
                refcode = entry.identifier
                mol = entry.molecule

                if TEST_MODE:
                    print(f"[{i}] Checking: {refcode} ...", end="")

                # --- A. Element filtering (Comprehensive metal detection) ---
                atoms = mol.atoms
                found_bad_element = None
                for atom in atoms:
                    if atom.atomic_symbol in EXCLUDE_ELEMENTS:
                        found_bad_element = atom.atomic_symbol
                        break

                if found_bad_element:
                    skipped_metal += 1
                    if TEST_MODE:
                        print(f" -> [Skip] Metal found: {found_bad_element}")
                    continue

                # --- B. Extract the largest component ---
                if mol.components and len(mol.components) > 1:
                    main_mol = max(mol.components, key=lambda m: len(m.atoms))
                    comp_info = f"Max component ({len(main_mol.atoms)}/{len(atoms)} atoms)"
                else:
                    main_mol = mol
                    comp_info = f"Single component ({len(atoms)} atoms)"

                # --- C. Check 3D coordinate completeness ---
                missing_coords = False
                for atom in main_mol.atoms:
                    if atom.coordinates is None:
                        missing_coords = True
                        break

                if missing_coords:
                    skipped_error += 1
                    if TEST_MODE:
                        print(f" -> [Skip] Missing 3D coordinates!")
                    continue

                # --- D. Save files ---

                # 1. Save CIF
                cif_path = os.path.join(CIF_OUT_DIR, f"{refcode}.cif")
                with io.EntryWriter(cif_path) as writer:
                    writer.write(entry)

                # 2. Save XYZ
                xyz_path = os.path.join(XYZ_OUT_DIR, f"{refcode}.xyz")
                with open(xyz_path, 'w', encoding='utf-8') as f:
                    f.write(f"{len(main_mol.atoms)}\n")
                    f.write(f"{refcode}\n")
                    for atom in main_mol.atoms:
                        x = atom.coordinates.x if atom.coordinates else 0.0
                        y = atom.coordinates.y if atom.coordinates else 0.0
                        z = atom.coordinates.z if atom.coordinates else 0.0
                        f.write(f"{atom.atomic_symbol:<4} {x:.6f} {y:.6f} {z:.6f}\n")

                # --- E. Write to CSV list ---
                f_csv.write(f"{refcode}\n")
                if saved_count % 1000 == 0:
                    f_csv.flush()

                saved_count += 1
                if TEST_MODE:
                    print(f" -> [OK] Saved to CSV. {comp_info}")

            except Exception as e:
                skipped_error += 1
                if TEST_MODE:
                    print(f" -> [Error] {e}")
                elif skipped_error % 1000 == 0:
                    print(f"Error processing {refcode}: {e}")
                continue

    # Final summary
    print("\n" + "=" * 40)
    print("Processing finished")
    print(f"Successfully extracted and recorded: {saved_count}")
    print(f"Skipped (contains metals): {skipped_metal}")
    print(f"Skipped (error/no coordinates): {skipped_error}")
    print(f"List file: {CSV_OUT_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    batch_process()
