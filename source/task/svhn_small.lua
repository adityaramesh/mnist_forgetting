require "torch"

function get_task_info()
        local pp_data_dir = "data/svhn/preprocessed"
        local train_file = paths.concat(pp_data_dir, "train_small.t7")
        local test_file = paths.concat(pp_data_dir, "test.t7")
        return torch.load(train_file), torch.load(test_file), 10
end
