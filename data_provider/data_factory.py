from .data_loader import (
    APAVALoader,
    ADFDLoader,
    ADFDDependentLoader,
    TDBRAINLoader,
    PTBLoader,
    PTBXLLoader,
)
from .uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    "APAVA": APAVALoader,
    "TDBRAIN": TDBRAINLoader,
    "ADFD": ADFDLoader,
    "ADFD-Sample": ADFDDependentLoader,
    "PTB": PTBLoader,
    "PTB-XL": PTBXLLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    _flag = str(flag).lower()
    # 禁用 VAL/TEST 集的打乱，确保与保存的不确定性数组索引严格对齐
    if _flag in ("test", "val"):
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.task_name == "classification":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader


