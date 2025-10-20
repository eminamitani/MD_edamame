import glob
from ase.io import read
from ase.io.trajectory import Trajectory
import os
from ase import Atoms
import re
from tqdm import tqdm
import argparse

def read_custom_xyz(filename):
    # 'A'や'B'を、aseが認識できるダミーの元素記号にマッピングする
    symbol_map = {'A': 'H', 'B': 'He'}

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"警告: ファイルが見つかりません: {filename}")
        return

    i = 0
    while i < len(lines):
        try:
            num_atoms = int(lines[i])
            frame_lines = lines[i+2 : i+2+num_atoms]
            
            if len(frame_lines) < num_atoms:
                break

            symbols = []
            positions = []
            for line in frame_lines:
                parts = line.split()
                original_symbol = parts[0]
                
                mapped_symbol = symbol_map.get(original_symbol, original_symbol)
                symbols.append(mapped_symbol)
                
                positions.append([float(p) for p in parts[1:4]])

            atoms = Atoms(symbols=symbols, positions=positions)
            yield atoms

            i += num_atoms + 2
        except (ValueError, IndexError):
            break

def main():
    #コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = ".xyzファイルのパス"
    )
    parser.add_argument(
        '--output', type = str, 
        default = "./trajectory.traj", 
        help =  "出力する.ptファイルのパス"
    )
    args = parser.parse_args()

    input_xyz_file = args.input
    output_traj_file = args.output

    traj = Trajectory(output_traj_file, "w")

    frames_generator = read_custom_xyz(input_xyz_file)

    for atoms_frame in tqdm(frames_generator): 
        traj.write(atoms_frame)

    traj.close()

if __name__ == "__main__":
    main()