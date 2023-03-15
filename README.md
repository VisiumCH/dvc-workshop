
# DVC Workshop


This repository features a code along with a workshop showcasing Visium's way of using DVC.


By the end of this workshop you will have learned how to :
- set up remote bucket for data versioning with DVC
- pull and push data with DVC
- set up DVC a pipeline
- create DVC stage


DVC architecture in the repo will also be tackled.


This workshop features a handwritten digits image classifier. The model that performs this multi-label image classification is kept simple for the sake of focusing on the tool presented.
<p align="middle">
<img src="https://github.com/VisiumCH/dvc-workshop/blob/tuto-complete/data/sample_images/two.png" width="100" />
<img src="https://github.com/VisiumCH/dvc-workshop/blob/tuto-complete/data/sample_images/five.png" width="100" />
<img src="https://github.com/VisiumCH/dvc-workshop/blob/tuto-complete/data/sample_images/nine.png" width="100" />
</p>



<details>
<summary><h2>1. Getting started : Environment</h2></summary>




### Install python 3.10 with pyenv


**Disclaimer** : If you already have Python __3.10__ installed locally you may skip this part.


This workshop will require you to have a working Python 3.10 installation and is enforced in our environment manager. In case you don't have any Python install (highly unlikely...) or other versions installed, we recommend using (drum roll ...) `pyenv`, to create a virtual environment, install and manage different Python versions. This is a standard used in Visium.


In case pyenv installation is too cumbersome, you can always revert the python version of the environment to the one you have locally.


To install this package, follow the [recommendations](https://github.com/pyenv/pyenv) according to your operating system. Though we suggest using the installer by running


```bash
curl https://pyenv.run | bash
```


Then, you can list all Python 3.10 releases available for install by running:


```bash
pyenv install --list | grep " 3.10"
```


and pick one to install running:


```bash
pyenv install <version>
```


and set it as default Python version:


```bash
pyenv global <version>
```


you can check at all times which versions are installed locally and which one is used with:


```bash
pyenv versions
```




## Manage your python environment


### Install pipenv




This workshop uses`pipenv` as package manager, if you don't have it installed, it is available with `pip` running:


```
pip install pipenv
```




### Set up the environment - Pipenv
The python environment is managed with `pipenv`. You can set up your environment with the following steps:

- Run `pipenv lock` to generate the `Pipfile.lock` which lists the version of your python packages.
- Run `pipenv install --dev` to actually create a virtual environment and install the python packages. The flag `--dev` allows you to install the development packages (for linting, ...).


`pipenv` relies upon the *Pipfile* to install the required packages. By default, we have specified a few libraries to install.

Take a look at the *Pipfile*. Can you figure out its logic?

You can install new libraries either by asking pipenv to do so:


```
pipenv install <package_name>
```


or by editing the *Pipfile* with your library:


```
<package_name> = <version>
```

by putting "*" in place of ```<version>```, you let `pipenv` manage the versions of all libraries to avoid conflict dependencies.


>#### 📓 Exercise
>
>We intentionally left out a tensorflow library to install, add the following to your environment :
>
>* tensorflow
>
>For Mac OS, install:
>
>
>* tensorflow-macos
>
>
>Finally, looking at the dev section, you might have guessed that those packages are here to aid during development. As such, `black` is a python code formatter, `isort` to order the imports, `pre-commit`to ensure no code is pushed with formating etc...
>
>
>adding the ```--dev``` during the install will allow you to use these libraries.

### Set up the environment - Google Cloud Plateform Credentials

During this workshop, we are going to be using storage services of Google Cloud Platform (GCP). To that end, we have created a service account, with an associate secret key. You will be using this service account to authenticate to GCP. You will find the key in the slack channel dedicated to the workshop. Store the key in a `json` file under the path `./.google_credentials.json`.

Next, we are going to store the path in the environment variable in a .env file. If it is not already there, create a .env file and write:

```
export GOOGLE_APPLICATION_CREDENTIALS=./.google_credentials.json
```



That way, upon environment creation with pipenv, the variable will be set properly and you will be able to communicate with the google cloud storage bucket.



### Activate and use the pipenv environment


To run code under your newly set up environment, activate it with 

```bash
pipenv shell
```

Note that this command will also source environment variables from your `.env` file.




### Pipenv and GitHub


Make sure to commit the `Pipfile.lock` in `git`. It will make your code more reproducible because other developers could install the exact same python packages as you used.


---


</details>

<details>

<summary><h2>2. Preprocess step:</h2></summary>

Now that we have properly set the environment, let's get in the nitty gritty of this workshop.


In this part, you are going to learn how to:


* Set up a DVC remote storage
* Understand DVC pipeline step
* Understand DVC data versioning workflow


and also brush up a bit on your... Python skills!


As you might have guessed, one of the reasons we use DVC at Visium when it comes to ML projects is its versatility, allowing us to version code, data, models but also to architect the execution pipeline.


We will come to that during this part, but first let's get our hands on some data!


To keep the focus on the tool rather than the problem, we've decided during this workshop to code a small image classification model that labels handwritten digits images in black and white (also known as mnist :)).


For preprocessing, your goal will be to standardize the images and apply some rotation ... In the Visium way!


That means, implementing a proper DVC step for preprocessing and versioning the output data of this step.


Now, we have set up a Google Storage (gs) Bucket where we store our data and later the different versioned files. As for a client project, two buckets are often created, here :


* ext-workshop-raw: keeping safe the original version of the data
* ext-dvc-workshop-cache: storing the different file hashes (versions)


First step will be to get the data from the first bucket. This will be the very first step of our DVC pipeline and a good example of how to set up one.




### A. DVC Pipeline

DVC Pipelines allows you to version control your code and track changes at all times. Pipelines are organized on steps with possible dependencies. As such if a step changes, DVC detects code edits and output changes of that given step and runs the all other dependent steps to reflect the modifications.


On the other hand, if a change in a given step has no incidence on other upstream ones, DVC will not re-run the complete pipeline all-over.

File hashes can be seen in the __dvc.lock__ file and pipeline steps in the __dvc.yaml__ one.


Steps have :
- a name : here same as the name of the directory.
- dependencies : directory and files used for running it
- outputs : files or directory where outputs are stored.
- parameters : usually user defined parameters the step depends on. Here contained in __dvc_workshop/params.py__.
- command : Python command for dvc to run the step.


With the cookiecutter architecture, steps are pre-defined, and you can adapt them with the appropriate changes directly in the __dvc.yaml__. DVC detects outputs of each step and pushes changes to the remote automatically.


To manually create a step, you can use:


```
dvc add stage -n <stage name> -d <dependencies> -p <parameters> -o <outputs> cmd
```
You can also checkout the file structure in the __Pipfile__ and implement the step manually.






#### B. Downloading data


To pull the data from the gs bucket, we have coded a small script on which we are going to be building out the data retrieving step. You can look up the script under __dvc_workshop/utils/generate_mnist_dataset.py__.
The objective of this step will be to execute this script, store data under __data/download_mnist__. As explained above, it is also important to mention the dependencies to keep track of the versions of, here the script and bucket raw data used. There are no particular parameters for this step.


Therefore, we execute the following command to create our first step:


```
dvc stage add -n download_mnist -d gs://ext-dvc-workshop-raw/mnist -d dvc_workshop/utils/generate_mnist_dataset.py -o data/download_mnist python -m dvc_workshop.utils.generate_mnist_dataset --output-image-path "data/download_mnist/Images" --output-df-path "data/download_mnist"
```


You can look up for yourself the result of this command by opening up the __dvc.yaml__ and spotting the changes...


#### C. Remote Bucket


Now, the second step before going right into the preprocessing is setting up a "remote" for DVC. By specifying a remote, we instruct DVC the locations where to store file versions. When tracking a new file (by specifying it as a dependence in step as seen above for example), DVC computes a hash of that file to fix its state. Later on, we would like to store all computed hashes in a remote location, other than your local cache, and more importantly shared. That way, if many developers work on the same project, that can all get access to the latest versions of the codes, models, data ... (without computing them!)


This is where the second gs bucket comes in handy. For instance, upon downloading the raw mnist data, we would like to version this data in case it gets corrupted in the future, we can always back track to it. You can think of it as a git repository, as a matter of fact, DVC is built on top of git, so a lot of the commands are similar. As such, when adding files to be tracked using


```
dvc add <file_name>
```

the destination of the corresponding files hashes when pushed:

```
dvc push
```
are sent to the remote.

Finally, you need to instantiate remote running:

```
dvc remote add -d <remote-name> gs://<bucket-name>/<folder-name>
```
with `<bucket-name>` being dvc-workshop-cache and `<folder-name>` your SCIPER number. You can name the remote as you please but let's agree on calling it dvc-workshop.


You can look at the effect of this command in the config file under __.dvc__.


#### D. Piecing it together:


You have successfully set up your first DVC step and declared a default remote storage location. Now you are going to test out this workflow.


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


#### E. Time to exercise:


It is TIME.


In this part, the objective is create a dvc preprocess step that:


1. loads data from previously downloaded
2. applies standardization, some rotation and cropping
3. saves the output images under ___data/preprocess___


and then push the resulting files to the remote.

We have already implemented the code structure for loading data, rotating and cropping in ___dvc_workshop/pipeline/preprocess.py___. Standardization was left empty intentionally for you to fill it.




Remember to have a look at the DAG once you have created your pipeline.

---

</details>


<details>
<summary><h2>3. Training step:</h2></summary>

In this step, you are going to be implementing and training a small multilabel classification model.


The idea is to understand how DVC handles the execution of the steps as they grow more numerous.


The model implemented has shown some overfitting, and we would like you to add some dropout layers to the mix to mitigate this effect. Head to the model folder
and checkout the instructions in ___dvc_workshop/models/classifier.py___.


We have implemented for you the complete training procedure under ___dvc_workshop/pipeline/train/training.py___.


As you might have guessed, your task will be to implement the training dvc step. You should have a look at the files above as well as some of their dependencies to figure out the different elements of your step.


At the end, your dag should look like the following:


<pre>
+----------------+
| download_mnist |
+----------------+
        *
        *
        *
  +------------+
  | preprocess |
  +------------+
        *
        *
        *
    +-------+
    | train |
    +-------+
</pre>


Don't forget to push the output of this step hashes to the remote.


Also, you can play around to better understand DVC's functioning for instance by deleting the local preprocess data and pulling it again from the remote. You can also rerun the pipeline after that, or modify some of the files before running it.


---
</details>

 
<details>
<summary><h2>4. Evaluation step:</h2></summary>

You should now have successfully trained your classifier and have it saved along with the training history. Let's evaluate the model on the test set you generated earlier.


Once again, your job here is to complete the code snippet in __dvc_workshop/pipeline/evaluate.py__ and right down the DVC step accordingly. Make sure to include all dependencies :-).


---

</details>



<details>
<summary><h2>5. Plotting step:</h2></summary>

In this part you are asked to complete the code for plotting the validation loss and accuracy of the training history. Remember to leverage DVC while respecting the code architecture to implement a new step in the pipeline. The resulting plots should be saved in the same manner other steps did.


Remember to look at existing functions to leverage and to push your results to the remote.

The resulting pipeline from all the above steps should look like the following:


<pre>
    +----------------+       
    | download_mnist |       
    +----------------+       
             *               
             *               
             *               
      +------------+         
      | preprocess |         
      +------------+         
        *        **          
      **           *         
     *              **       
+-------+              *      
| train |*             *      
+-------+ ***          *      
   *        ***       *      
   *           ***    *      
   *              **  *      
+------+         +----------+ 
| plot |         | evaluate | 
+------+         +----------+ 
</pre>


</details>



