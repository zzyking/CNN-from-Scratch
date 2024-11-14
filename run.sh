model_types=("simple" "deep")
normalizations=("" "batch" "layer")
data_augmentations=("True" "False")
weight_decays=("0.0" "0.0001" "0.001" "0.01")
dropout_rates=("0.0" "0.2" "0.5" "0.8")

for model_type in "${model_types[@]}"; do
  for normalization in "${normalizations[@]}"; do
    for data_augmentation in "${data_augmentations[@]}"; do
      for weight_decay in "${weight_decays[@]}"; do
        for dropout_rate in "${dropout_rates[@]}"; do

          command="python main.py --model $model_type"

          if [ -n "$normalization" ]; then
            command+=" --normalization $normalization"
          fi

          if [ "$data_augmentation" == "True" ]; then
            command+=" --data_augmentation"
          fi

          command+=" --weight_decay $weight_decay"

          command+=" --dropout_rate $dropout_rate"

          echo "Running: $command"

          $command
        done
      done
    done
  done
done
