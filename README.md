# info8010_project

- for now the app only manages 10 classes :
airplane, apple, bicycle, car, cat, cloc;, dog, face, house & tree

- requirements.txt does not include torch and torchvision so you can use the version you want CPU/GPU

scripts
- load_dataset.py : loads dataset

src
- train.py : trains and saves a simple cnn in models/simple_cnn.pt

launch flask app
- python -m webapp.app