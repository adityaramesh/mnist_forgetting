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
