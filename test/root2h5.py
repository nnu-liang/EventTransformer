import uproot
import h5py
import numpy as np
import argparse

def root_to_h5_variable_length(root_file_path, h5_file_path):
    # 打开ROOT文件
    with uproot.open(root_file_path) as root_file:
        # 读取固定名称的tree
        tree = root_file["tree"]
        
        # 打开/创建HDF5文件
        with h5py.File(h5_file_path, 'w') as h5_file:
            # 遍历tree中的所有branch
            for branch_name, branch in tree.items():
                # 获取branch的数据
                data = branch.array(library="np")
                
                if isinstance(data[0], np.ndarray):
                    # 如果是vector类型（例如vector<float>或vector<int>），创建可变长度的HDF5数据集
                    dt = h5py.vlen_dtype(np.dtype(data[0].dtype))
                    h5_file.create_dataset(branch_name, data=data, dtype=dt, compression="gzip")
                elif isinstance(data[0], (np.float32, np.float64, np.int32, np.int64)):
                    # 如果是int或float类型，直接存储
                    h5_file.create_dataset(branch_name, data=data, dtype=data.dtype, compression="gzip")
                elif isinstance(data[0], (bool, np.bool_)):
                    # 如果是bool类型，将其转换为0和1来存储
                    h5_file.create_dataset(branch_name, data=data.astype(np.int8), dtype=np.int8, compression="gzip")
                else:
                    print(f"Unsupported branch type for {branch_name}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Convert a ROOT file to HDF5 format.")
    parser.add_argument("root_file_path", type=str, help="Path to the input ROOT file")
    parser.add_argument("h5_file_path", type=str, help="Path to the output HDF5 file")
    
    args = parser.parse_args()
    
    # 调用转换函数
    root_to_h5_variable_length(args.root_file_path, args.h5_file_path)

if __name__ == "__main__":
    main()

