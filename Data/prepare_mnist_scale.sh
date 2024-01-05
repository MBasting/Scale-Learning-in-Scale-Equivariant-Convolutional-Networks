# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

MNIST_DIR="${MNIST_DIR:-./}"
MNIST_SCALE_DIR="${MNIST_SCALE_DIR:-./}"

echo "Preparing datasets..."
for i in {0..5}
do 
    echo ""
    echo "Dataset [$((i+1))/6]"
    python prepare_datasets_org.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --gauss Middle --cauchy True
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --gauss Small
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --gauss Large
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --small_test True
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --equi True
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 0.3 --download --seed $i --equi True --small_test True
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 1.0 --download --seed $i --single_scale_train True
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 1.0 --download --seed $i --single_scale_train True --val_size 5000
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 1.0 --download --seed $i --single_scale_train True --val_size 5000 --img_size 160

    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 1.0 --download --seed $i --single_scale_train True --resize_factor 2.0 --val_size 5000
    # python prepare_datasets.py --source "$MNIST_DIR" --dest "$MNIST_SCALE_DIR" --min_scale 1.0 --download --seed $i --single_scale_train True --resize_factor 4.0

done

echo "All datasets are generated and validated."
