import cbor2
import torch
import os

# normalization function from the ASIC LSTM code
def normalize_data_std(train_data, test_data):
    mean_data = torch.mean(torch.cat((train_data.flatten(), test_data.flatten())))
    std_data = torch.std(torch.cat((train_data.flatten(), test_data.flatten())))
    #train_data = train_data / (max_data - min_data)
    #test_data = test_data / (max_data - min_data)

    train_data = (train_data - mean_data) / (std_data)
    train_data = 2 * train_data / torch.max(train_data)
    test_data = (test_data - mean_data) / (std_data)
    test_data = 2 * test_data / torch.max(test_data)

    #print(torch.max(train_data), torch.min(train_data))
    #print(torch.max(test_data), torch.min(test_data))
    print("mean-std:", mean_data, std_data)

    return train_data, test_data
# function ens

# hardcoded setup paths
train_in = r"C:\Users\Laura\Downloads\gait-model-data\training"
test_in = r"C:\Users\Laura\Downloads\gait-model-data\testing"

train_out = r"C:\Users\Laura\Downloads\Normalized_Data\training"
test_out = r"C:\Users\Laura\Downloads\Normalized_Data\testing"

os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

# data containers
train_acc_list, train_gyr_list = [], []
test_acc_list, test_gyr_list = [], []
file_metadata = [] 

print("Phase 1: Harvesting data...")
# the data files are downloaded from Edge Impulse so they are cbor files
for input_dir, output_dir in [(train_in, train_out), (test_in, test_out)]:
    is_train = (input_dir == train_in)
    files = [f for f in os.listdir(input_dir) if f.endswith('.cbor')]
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'rb') as f:
            data = cbor2.load(f)
        
        values = data['payload']['values']
        sensors = data['payload']['sensors']
        
        acc_idx = [i for i, s in enumerate(sensors) if 'acc' in (s['name'] if isinstance(s, dict) else s).lower()]
        gyr_idx = [i for i, s in enumerate(sensors) if 'gyr' in (s['name'] if isinstance(s, dict) else s).lower()]

        file_metadata.append({
            'filename': filename,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'is_train': is_train,
            'row_count': len(values),
            'acc_idx': acc_idx,
            'gyr_idx': gyr_idx
        })

        for row in values:
            if is_train:
                for idx in acc_idx: train_acc_list.append(row[idx])
                for idx in gyr_idx: train_gyr_list.append(row[idx])
            else:
                for idx in acc_idx: test_acc_list.append(row[idx])
                for idx in gyr_idx: test_gyr_list.append(row[idx])

# calculate normalization
print("\nNormalizing Accelerometers:")
tr_acc_norm, ts_acc_norm = normalize_data_std(
    torch.tensor(train_acc_list, dtype=torch.float32),
    torch.tensor(test_acc_list, dtype=torch.float32)
)

print("\nNormalizing Gyroscopes:")
tr_gyr_norm, ts_gyr_norm = normalize_data_std(
    torch.tensor(train_gyr_list, dtype=torch.float32),
    torch.tensor(test_gyr_list, dtype=torch.float32)
)

# convert tensors to iterators for efficient redistribution
# this allows us to "pop" values one by one in the same order we harvested them
iterators = {
    'train_acc': iter(tr_acc_norm.tolist()),
    'test_acc': iter(ts_acc_norm.tolist()),
    'train_gyr': iter(tr_gyr_norm.tolist()),
    'test_gyr': iter(ts_gyr_norm.tolist())
}

# save
print(f"\nPhase 2: Saving files...")

for meta in file_metadata:
    original_path = os.path.join(meta['input_dir'], meta['filename'])
    with open(original_path, 'rb') as f:
        data_dict = cbor2.load(f)
    
    values = data_dict['payload']['values']
    new_values = []
    
    acc_iter = iterators['train_acc'] if meta['is_train'] else iterators['test_acc']
    gyr_iter = iterators['train_gyr'] if meta['is_train'] else iterators['test_gyr']

    for r in range(meta['row_count']):
        new_row = list(values[r])
        
        for idx in meta['acc_idx']:
            new_row[idx] = next(acc_iter)
        for idx in meta['gyr_idx']:
            new_row[idx] = next(gyr_iter)
            
        new_values.append(new_row)

    data_dict['payload']['values'] = new_values
    save_path = os.path.join(meta['output_dir'], meta['filename'])
    
    with open(save_path, 'wb') as f:
        cbor2.dump(data_dict, f)
    
    print(f"  Saved: {os.path.join(os.path.basename(meta['output_dir']), meta['filename'])}")

print(f"\nSUCCESS: Global normalization applied and files saved.")
