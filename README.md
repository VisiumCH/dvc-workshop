# DVC Workshop

This repository features a code along workshop showcasing Visium's way of using DVC.

By the end of this workshop you will have learned how to : 
  - set up remote bucket for data versioning with DVC
  - pull and push data with DVC
  - set up DVC a pipeline
  - create DVC stage

DVC architecture in the repo will also be tackled.

This workshop features a movie poster classifier. The code originates from a synergie with another internal project. The model performs a multi-label image classification. 
<p align="middle">
  <img src="https://github.com/laxmimerit/Movies-Poster_Dataset/blob/master/Images/tt0085271.jpg" width="100" />
  <img src="https://github.com/laxmimerit/Movies-Poster_Dataset/blob/master/Images/tt5022418.jpg" width="100" /> 
  <img src="https://github.com/laxmimerit/Movies-Poster_Dataset/blob/master/Images/tt4288636.jpg" width="100" />
</p>
Checkout the [repository](https://github.com/VisiumCH/image-classification-autotrain) of the project for more information.



## 1. Getting started : Environment

### Set up python environment

**Disclaimer** : If you already have Python __3.10__ installed locally you may skip this part. 

This worksop will require you to have a working Python 3.10 installation. In case you don't have any Python instal (highly unlikely...) or other versions installesd, we recommend using (drum roll ...) `pyenv`, to create virtual environment, install and manage different Python versions. 

To install this package, follow the [recommendations](https://github.com/pyenv/pyenv) according to your operating system. Though we suggest using the installer by running

```
curl https://pyenv.run | bash

```

Then, you can list all Python 3.10 releases available for install by running: 

```
pyenv install --list | grep " 3.10"

```

and pick one to install running: 

```
pyenv install <version>

```

and set it as default Python version:

```
pyenv global <version>
```

you can check at all times which versions are installed locally and which one is used with :

```
pyenv versions
```


## Manage your python environment

### Requirements: 


This workhop uses `pipenv`as package manager, if you don't have it installed, it is available with `pip` running: 

```
pip install pipenv

```


### Set up the environment
The python environment is managed with `pipenv`. You can set up your environment with the following steps:

- Run `pipenv lock`to generate the `Pipfile.lock` which lists the version of your python packages.
- Run `pipenv install --dev` to actually create a virtual environment and install the python packages. The flag `--dev` allows to install the development packages (for linting, ...).

`pipenv` relies upon the *Pipfile* to install the required packages. By default, we have specified few libraries to install. 

Take a look at the *Pipfile*, can you figure out its logic? 

You can install new libraries either by asking pipenv to do so: 

```
pipenv install <package_name>
```

or by editing the *Pipfile* with your library: 

```
<packge_name> = <version>
```

by puting "*" inplace of <version>, you let `pipenv` manage the versions of all librabries to avoid confilct dependencies. 

### Exercise 

We intentionally left out a fiew dependencies to install, add the following to your environment : 

**TBD List Packages**


Finally, looking at the dev section, you might have guessed that thoes packages are here to aid during development. As such, `black`is a python code formatter, `isort` to order the imports, `pre-commit`to ensure no code is pushed with formating etc... 

adding the ```--dev``` during the install will allow you to use these libraries. 

### Activate and use your environment

To run code under your newly set up environment, you have two options:

- *Open a shell*: activate your environment with `pipenv shell`. Note that this command will also source environment variables from your `.env` file.

- *Pipenv CLI*: you can also run scripts using your python environment with `pipenv run script.py`. This can be convenient within a `docker build` execution for example.


### Some tips about pipenv

**About deploying in production**

Note that when deploying your code in production, you should not install the dev package, it is preferred to run the following command: `pipenv install --system --deploy`.

**About using git with pipenv**

Make sure to commit the `Pipfile.lock` in `git`. It will make your code more reproducible because other developers could install the exact same python packages as you used.


---
