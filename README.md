## Traffic Sign Classifier (Self-Driving Car Engineer Nanodegree)

This repository contains code, source and resuling images and videos
with lane lines detected. The code is in Jupyter notebook
[`Traffic_Sign_Classifier.ipynb`](./Traffic_Sign_Classifier.ipynb). Please, follow
the instructions below to prepare an environment to run this notebook.
There is also [`Writeup.md`](./Writeup.md) where the pros and cons of the implementation described.

#### Preparing Environment
In order to provide all the dependencies to the code in Jupyter notebook,
build a docker image with the following command:
```bash
docker build --tag udacity/carnd-term1-starter-kit:patched --file Dockerfile .
```
Then start a docker container with:
```bash
docker run --interactive --tty --rm --publish 8888:8888 --volume $PWD:/src udacity/carnd-term1-starter-kit:patched
```
Note, that the current directory will become your working directory in Jupyter notebook.

Then, copy a link from the console to your browser and start exploring
the source code. The link will be similar but not equal to
`http://localhost:8888/?token=eb26e4a2b935c384dc3e0230a8181984f07da6be9df0c1b8`.

#### Notice
Dockerfile and some functions from [`Traffic_Sign_Classifier.ipynb`](./Traffic_Sign_Classifier.ipynb) are
provided by [Udacity.com](https://www.udacity.com).

