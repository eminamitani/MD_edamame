import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import atomic_masses, atomic_numbers

def create_random_sio2(density_g_cm3, num_si):
    """
    指定された密度とSi原子数で、SiとOを1:2の割合でランダムに配置した
    アモルファスSiO2構造を生成します。

    Args:
        density_g_cm3 (float): 構造の目標密度 (単位: g/cm^3)
        num_si (int): 構造に含めるSi原子の数
    """
    # --- 1. 原子数の設定 ---
    num_o = 2 * num_si
    total_atoms = num_si + num_o
    symbols = ['Si'] * num_si + ['O'] * num_o

    # --- 2. セル体積の計算 ---
    # 各原子の質量 (atomic mass units, u) を取得
    mass_si = atomic_masses[atomic_numbers['Si']]
    mass_o = atomic_masses[atomic_numbers['O']]

    # 系全体の総質量 (g) を計算
    # 1 u = 1 / (N_A) g (N_A: アボガドロ定数)
    total_mass_u = num_si * mass_si + num_o * mass_o
    avogadro_number = 6.02214076e23
    total_mass_g = total_mass_u / avogadro_number

    # 密度から体積 (cm^3) を計算
    volume_cm3 = total_mass_g / density_g_cm3

    # 体積を cm^3 から Å^3 (オングストローム^3) に変換 (1 cm = 1e8 Å)
    volume_A3 = volume_cm3 * (1e8)**3

    # 立方体セルの辺の長さ (Å) を計算
    cell_length = volume_A3**(1/3.0)
    cell = [cell_length, cell_length, cell_length]

    # --- 3. 原子のランダム配置 ---
    # scaled_positionsは0から1の範囲で座標を指定するため、セルサイズに依存しない
    scaled_positions = np.random.rand(total_atoms, 3)

    # Atomsオブジェクトの作成
    atoms = Atoms(symbols=symbols,
                  scaled_positions=scaled_positions,
                  cell=cell,
                  pbc=True) # 周期境界条件を有効にする

    # --- 4. 情報表示とファイル出力 ---
    # 実際の密度を計算して確認 (Å^3 と g/cm^3 の変換)
    actual_volume_A3 = atoms.get_volume()
    actual_volume_cm3 = actual_volume_A3 / (1e8)**3
    actual_density = total_mass_g / actual_volume_cm3

    print("--- 構造生成サマリー ---")
    print(f"入力された密度: {density_g_cm3:.3f} g/cm^3")
    print(f"Si原子数: {num_si}")
    print(f"O 原子数: {num_o}")
    print(f"合計原子数: {total_atoms}")
    print(f"セルの一辺の長さ: {cell_length:.3f} Å")
    print(f"セルの体積: {actual_volume_A3:.3f} Å^3")
    print(f"計算された密度: {actual_density:.3f} g/cm^3")
    print("-------------------------")

    # ファイルに書き出し
    output_filename = 'random_sio2.xyz'
    write(output_filename, atoms, format='extxyz')
    print(f"構造を '{output_filename}' に保存しました。")


if __name__ == '__main__':
    try:
        # --- ユーザー入力 ---
        # 例として一般的な非晶質シリカの密度 2.2 g/cm^3 をデフォルト値とする
        input_density = float(input("目標密度を入力してください (g/cm^3) [例: 2.2]: ") or 2.2)
        input_num_si = int(input("Si原子の数を入力してください [例: 100]: ") or 100)

        # 関数を実行
        create_random_sio2(input_density, input_num_si)

    except ValueError:
        print("エラー: 数値を正しく入力してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")