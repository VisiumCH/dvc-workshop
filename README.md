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

## 2. Preprocess step: 

Now that we have propperly set the environment, let's get in the nitty gritty of this workshop. 

In this part, you are going to learn how to: 

* Set up a DVC remote storage
* Understand DVC pipeline step
* Understand DVC data versioning workflow 

and also brush up a bit on your... Python skills! 

As you might have guessed, one of the reason we use DVC at Visium when it comes to ML projects is its versatiliy, allowing to version code, data, models but also to architect the exection pipeline. 

We will come to that during this part, but first let's get our hands on some data! 

To keep the focus on the tool rather than the problem, we've decided during this workshop to code a small image classification model that labels hand written digits images in black and white (also known as mnist :)). 

For preprocessing, your goal will be to standardize the images and apply some rotation ... In the Visium way! 

That means, implementing a proper DVC step for preprocessing and versionning the output data of this step. 

Now, we have set up a Google Storage (gs) Bucket where we store our data and later the different versioned files. As for a client project, two buckets are often created, here : 

* dvc-workshop-raw: keeping safe the original version of the data 
* dvc-workshop-cache: storing the different file hashes (versions)


First step will be to get the data from the first bucket. This wil be the very first step of our DVC pipeline and a good example of how to setup one. 


###  A. DVC Pipeline

DVC Pipelines allows to version control your code and track changes at all times. Pipelines are organized on steps with possible dependencies. As such if a step changes, DVC detects code edits and output changes of that given step and runs the all other dependent steps to reflect the modifications.

On the opposite, if a change in a given step has no incidence on other upstream ones, DVC will not re-run the complete pipeline all-over.

File hashes can be seen in the __dvc.lock__ file and pipeline steps in the __dvc.yaml__ one.

Steps have :
- a name : here same as the name of the directory.
- dependencies : directory and files used for running it
- outputs : files or directory where output are stored.
- parameters : usually user defined parameters the step depends on. Here contained in __dvc_workshop/params.py__.
- command : Python command for dvc to run the step.

With the cookiecutter architecture, steps are pre-defined, and you can adapt them with the appropriate changes directly in the  __dvc.yaml__. DVC detects outputs of each steps and pushes changes to the remote automatically.

To manually create a step, you can use:

```
dvc add stage -n <stage name> -d <dependencies> -p <parameters> -o <outputs> cmd
```




#### B. Downloading data 

To pull the data from the gs bucket, we have coded a small script on which we are going to be building out data retreiving step. You can look up the script under __dvc_workshop/utils/generate_mnist_dataset.py__. 
The objective of this step will be to excecute this script, store data under __data/download_mnist__. As explained above, it is also important to mention the dependencies to keep track of the versions of, here the script and bucket raw data used. There are no particular parameter for this step.

Therefore, we execute the following command to create our first step: 

```
dvc stage add -n download_mnist -d gs://dvc-workshop-raw/mnist -d dvc_workshop/utils/generate_mnist_dataset -o data/download_mnist python -m dvc_workshop.utils.generate_mnist_dataset --output-image-path "data/download_mnist/Images" --output-df-path "data/download_mnist"


```

You can lookup for yourself the result of this command by opening up the __dvc.yaml__ and spotting the changes... 

#### C. Remote Bucket

Now, the second step before going right into the preprocessing is setting up a "remote" for DVC. By specifying a remote, we instruct DVC the locations where to store file versions. When tracking a new file (by specifying it as a dependence in step as seen above for example), DVC comptes a hash of that file to fix its state. Later on, we would like to store all computed hashes in a remote location, other than your local cache, and more importantly shared. That way, if many developers work on the same project, that can all get access to the latest versions of the codes, models, data ... (without computing them!)

This is where the second gs bucket comes in handy. For instance, upon downloading the raw mnist data, we would like to version this data in case it gets corrupted in the future, we can always back track to it. You can think of it as git repository, as a matter of fact, DVC is built on top of git, so a lot of the commands are similar. As such, when adding files to be tracked using 

```
dvc add <file_name>

```
the destination of the corresponding files hashes when pushed: 
```
dvc push
```
are sent to the remote. 

So now that we now why use it, let's set it up :) 

To instantiate remote, run: 

```
dvc remote add -d <remote-name> gs://<bucket-name>/<folder-name>
```
with `<bucket-name>`being dvc-workshop-cache. You can name the remote as you please but let's agree on calling it dvc-workshop. 

#### D. Piecing it together: 

You have successfuly set up your first DVC step and declared a default remote storage location. Now you are going to test out this workflow. 

With dvc you can first visualize your pipeline by running: 

```
dvc dag
```

this will display the directed acyclic graph, featuring your steps and the dependencies among each of them. 

For now, the DAG displayed only contains one step but we will get back to that shortly. 

To execute the pipeline, you can run: 

```
dvc repro
````
This reproduces the steps in order accounting for changes if any. 

Last but not least, dependencies, parameters and outputs in the pipeline are automatically versioned (no need to run ```dvc add <file>```) upon execution. The last step will be to push the generated hashes back to the remote we set up! 
ß
#### E. Time to exercise:

It is TIME. 

In this part, the objective is create a dvc preprocess step that: 

1. loads data from previously downloaded
2. applies standardization, some roatation and cropping
3. saves the output images under ___data/preprocess___

and then push the resulting files to the remote.

We have already implemented the code structure for loading data, rotating and cropping in ___dvc_workshop/pipeline/preprocess.py___. Standardization was left empty intentionnaly for you to fill it.


Remember to have a look a at the DAG once you have created your pipeline.ß