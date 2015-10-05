require "rake/clean"

task :train_half_fixed_models do
	max_epochs = [10, 20, 30]
	pretrained_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80]

	max_epochs.each do |m|
		pretrained_epochs.each do |p|
			# Fix the first half of the weights.
			sh "th source/drivers/cnn_3x3_fix_params.lua " \
				"-max_epochs #{m} "                    \
				"-pretrained_epochs #{p} "             \
				"-half 1 "                             \
				"-task replace "                       \
				"-model cnn_3x3_#{p}_max_#{m}_2nd_half"

			# Fix the second half of the weights.
			sh "th source/drivers/cnn_3x3_fix_params.lua " \
				"-max_epochs #{m} "                    \
				"-pretrained_epochs #{p} "             \
				"-half 2 "                             \
				"-task replace "                       \
				"-model cnn_3x3_#{p}_max_#{m}_1st_half"
		end
	end
end

task :train_fused_models do
	max_epochs = [10, 20, 30]
	pretrained_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80]

	max_epochs.each do |m|
		pretrained_epochs.each do |p|
			ldir = "models/cnn_3x3_#{p}_max_#{m}_1st_half"
			rdir = "models/cnn_3x3_#{p}_max_#{m}_2nd_half"
			sh "th source/drivers/cnn_3x3_seeded_v1.lua " \
				"-max_epochs 80 "                     \
				"-task replace "                      \
				"-model cnn_3x3_#{p}_max_#{m}_fused " \
				"-left_half_dir #{ldir} "             \
				"-right_half_dir #{rdir}"
		end
	end
end

task :train_one_shot_model do
	sh "th source/drivers/train_full_v2.lua " \
		"-max_epochs 200 "                \
		"-task replace "                  \
		"-model cnn_3x3_full_v2"
end

task :scale_mnist do
	sh "mkdir -p data/mnist/scaled"

	sh "th source/drivers/scale_mnist.lua "     \
		"data/mnist/raw/train_32x32.t7 "    \
		"data/mnist/scaled/train_32x32.t7 "

	sh "th source/drivers/scale_mnist.lua "    \
		"data/mnist/raw/test_32x32.t7 "    \
		"data/mnist/scaled/test_32x32.t7 "
end

task :partition_mnist do
	sh "mkdir -p data/mnist/partitioned"

	sh "th source/drivers/partition_mnist.lua "      \
		"data/mnist/scaled/train_32x32.t7 "      \
		"data/mnist/partitioned/train_left.t7 "  \
		"data/mnist/partitioned/train_right.t7 "

	sh "th source/drivers/partition_mnist.lua "     \
		"data/mnist/scaled/test_32x32.t7 "      \
		"data/mnist/partitioned/test_left.t7 "  \
		"data/mnist/partitioned/test_right.t7 "
end

task :train_mnist_half_models do
	sh "mkdir -p models"

	# Train the left half.
	sh "th source/drivers/run_mnist_half.lua "                  \
		"-train_file data/mnist/partitioned/train_left.t7 " \
		"-test_file data/mnist/partitioned/test_left.t7 "   \
		"-max_epochs 50 "                                   \
		"-zero_half 2 "                                     \
		"-model left_half "                                 \
		"-task replace "

	# Train the right half.
	sh "th source/drivers/run_mnist_half.lua "                   \
		"-train_file data/mnist/partitioned/train_right.t7 " \
		"-test_file data/mnist/partitioned/test_right.t7 "   \
		"-max_epochs 50 "                                    \
		"-zero_half 1 "                                      \
		"-model right_half "                                 \
		"-task replace "
end

task :train_mnist_baseline_models do
	sh "mkdir -p models"

	# Train the baseline on the first task (digits 0--4).
	sh "th source/drivers/run_mnist.lua "                       \
		"-train_file data/mnist/partitioned/train_left.t7 " \
		"-test_file data/mnist/partitioned/test_left.t7 "   \
		"-model_file source/models/cnn_5x5_mnist_half.lua " \
		"-max_epochs 50 "                                   \
		"-model task_1 "                                    \
		"-task replace "

	# Train the baseline on the second task (digits 5--9).
	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_right.t7 "       \
		"-test_file data/mnist/partitioned/test_right.t7 "         \
		"-model_file source/models/cnn_5x5_mnist_half_seeded.lua " \
		"-max_epochs 50 "                                          \
		"-model task_2 "                                           \
		"-task replace "
end

task :train_mnist_fused_model do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist_fused.lua "                 \
		"-train_file data/mnist/partitioned/train_left.t7 " \
		"-test_file data/mnist/partitioned/test_left.t7 "   \
		"-left_half_dir models/left_half "                  \
		"-right_half_dir models/right_half "                \
		"-max_epochs 50 "                                   \
		"-model fused "                                     \
		"-task replace "
end

task :train_mnist_full_task_model do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_left.t7 "        \
		"-test_file data/mnist/partitioned/test_left.t7 "          \
		"-model_file source/models/cnn_5x5_mnist_half_seeded_2.lua " \
		"-max_epochs 50 "                                          \
		"-model full_task "                                        \
		"-task replace "
end

task :partition_mnist_16 do
	sh "mkdir -p data/mnist/partitioned"

	sh "th source/drivers/partition_mnist.lua "      \
		"data/mnist/scaled/train_16x16.t7 "      \
		"data/mnist/partitioned/train_left_16x16.t7 "  \
		"data/mnist/partitioned/train_right_16x16.t7 "

	sh "th source/drivers/partition_mnist.lua "     \
		"data/mnist/scaled/test_16x16.t7 "      \
		"data/mnist/partitioned/test_left_16x16.t7 "  \
		"data/mnist/partitioned/test_right_16x16.t7 "
end

task :train_mnist_half_models_16 do
	sh "mkdir -p models"

	# Train the left half.
	sh "th source/drivers/run_mnist_half.lua "                  \
		"-train_file data/mnist/partitioned/train_left_16x16.t7 " \
		"-test_file data/mnist/partitioned/test_left_16x16.t7 "   \
		"-max_epochs 30 "                                   \
		"-zero_half 2 "                                     \
		"-model left_half_16x16 "                                 \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_half_16x16.lua "

	# Train the right half.
	sh "th source/drivers/run_mnist_half.lua "                   \
		"-train_file data/mnist/partitioned/train_right_16x16.t7 " \
		"-test_file data/mnist/partitioned/test_right_16x16.t7 "   \
		"-max_epochs 30 "                                    \
		"-zero_half 1 "                                      \
		"-model right_half_16x16 "                                 \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_half_16x16.lua "
end

task :train_mnist_baseline_models_16 do
	sh "mkdir -p models"

	# Train the baseline on the first task (digits 0--4).
	sh "th source/drivers/run_mnist.lua "                       \
		"-train_file data/mnist/partitioned/train_left_16x16.t7 " \
		"-test_file data/mnist/partitioned/test_left_16x16.t7 "   \
		"-model_file source/models/cnn_5x5_mnist_half_16x16.lua " \
		"-max_epochs 30 "                                   \
		"-model task_1_16x16 "                                    \
		"-task replace "

	# Train the baseline on the second task (digits 5--9).
	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_right_16x16.t7 "       \
		"-test_file data/mnist/partitioned/test_right_16x16.t7 "         \
		"-model_file source/models/cnn_5x5_mnist_half_seeded_16x16.lua " \
		"-max_epochs 30 "                                          \
		"-model task_2_16x16 "                                           \
		"-task replace "
end

task :train_mnist_fused_model_16 do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist_fused.lua "                 \
		"-train_file data/mnist/partitioned/train_left_16x16.t7 " \
		"-test_file data/mnist/partitioned/test_left_16x16.t7 "   \
		"-left_half_dir models/left_half_16x16 "                  \
		"-right_half_dir models/right_half_16x16 "                \
		"-max_epochs 30 "                                   \
		"-model fused_16x16 "                                     \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_fused_16x16.lua "
end

task :train_mnist_full_task_model_16 do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_left_16x16.t7 "        \
		"-test_file data/mnist/partitioned/test_left_16x16.t7 "          \
		"-model_file source/models/cnn_5x5_mnist_half_seeded_16x16_2.lua " \
		"-max_epochs 30 "                                          \
		"-model full_task_16x16 "                                        \
		"-task replace "
end

task :partition_mnist_8 do
	sh "mkdir -p data/mnist/partitioned"

	sh "th source/drivers/partition_mnist.lua "      \
		"data/mnist/scaled/train_8x8.t7 "      \
		"data/mnist/partitioned/train_left_8x8.t7 "  \
		"data/mnist/partitioned/train_right_8x8.t7 "

	sh "th source/drivers/partition_mnist.lua "     \
		"data/mnist/scaled/test_8x8.t7 "      \
		"data/mnist/partitioned/test_left_8x8.t7 "  \
		"data/mnist/partitioned/test_right_8x8.t7 "
end

task :train_mnist_half_models_8 do
	sh "mkdir -p models"

	# Train the left half.
	sh "th source/drivers/run_mnist_half.lua "                  \
		"-train_file data/mnist/partitioned/train_left_8x8.t7 " \
		"-test_file data/mnist/partitioned/test_left_8x8.t7 "   \
		"-max_epochs 30 "                                   \
		"-zero_half 2 "                                     \
		"-model left_half_8x8 "                                 \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_half_8x8.lua "

	# Train the right half.
	sh "th source/drivers/run_mnist_half.lua "                   \
		"-train_file data/mnist/partitioned/train_right_8x8.t7 " \
		"-test_file data/mnist/partitioned/test_right_8x8.t7 "   \
		"-max_epochs 30 "                                    \
		"-zero_half 1 "                                      \
		"-model right_half_8x8 "                                 \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_half_8x8.lua "
end

task :train_mnist_baseline_models_8 do
	sh "mkdir -p models"

	# Train the baseline on the first task (digits 0--4).
	sh "th source/drivers/run_mnist.lua "                       \
		"-train_file data/mnist/partitioned/train_left_8x8.t7 " \
		"-test_file data/mnist/partitioned/test_left_8x8.t7 "   \
		"-model_file source/models/cnn_5x5_mnist_half_8x8.lua " \
		"-max_epochs 30 "                                   \
		"-model task_1_8x8 "                                    \
		"-task replace "

	# Train the baseline on the second task (digits 5--9).
	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_right_8x8.t7 "       \
		"-test_file data/mnist/partitioned/test_right_8x8.t7 "         \
		"-model_file source/models/cnn_5x5_mnist_half_seeded_8x8.lua " \
		"-max_epochs 30 "                                          \
		"-model task_2_8x8 "                                           \
		"-task replace "
end

task :train_mnist_fused_model_8 do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist_fused.lua "                 \
		"-train_file data/mnist/partitioned/train_left_8x8.t7 " \
		"-test_file data/mnist/partitioned/test_left_8x8.t7 "   \
		"-left_half_dir models/left_half_8x8 "                  \
		"-right_half_dir models/right_half_8x8 "                \
		"-max_epochs 30 "                                   \
		"-model fused_8x8 "                                     \
		"-task replace " \
		"-model_file source/models/cnn_5x5_mnist_fused_8x8.lua "
end

task :train_mnist_full_task_model_8 do
	sh "mkdir -p models"

	sh "th source/drivers/run_mnist.lua "                              \
		"-train_file data/mnist/partitioned/train_left_8x8.t7 "        \
		"-test_file data/mnist/partitioned/test_left_8x8.t7 "          \
		"-model_file source/models/cnn_5x5_mnist_half_seeded_8x8_2.lua " \
		"-max_epochs 30 "                                          \
		"-model full_task_8x8 "                                        \
		"-task replace "
end
