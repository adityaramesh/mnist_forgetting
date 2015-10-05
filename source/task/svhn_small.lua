require "torch"

function get_task_info()
        local pp_data_dir = "data/svhn/image_normalized"
        local train_file = paths.concat(pp_data_dir, "train_small_yuv.t7")
        local test_file = paths.concat(pp_data_dir, "test_yuv.t7")
        return torch.load(train_file), torch.load(test_file), 10
end
