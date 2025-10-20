from ase.io import read, write

type_map = {
    1: 'Na',
    2: 'O',
    3: 'Si',
}

input_file = 'data/sample_NS22_5.data'
output_file = 'data/NS22_5.xyz'

try:
    atoms = read(input_file, format='lammps-data')

    atom_types = atoms.get_array('type')
    
    chemical_symbols = [type_map[t] for t in atom_types]
    
    atoms.set_chemical_symbols(chemical_symbols)

    write(output_file, atoms)

except KeyError as e:
    print(f"❌ エラー: type_mapに定義されていない原子type {e} が見つかりました。")
    print("スクリプト冒頭の type_map に、お使いの原子typeと元素記号の対応をすべて記述してください。")
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")