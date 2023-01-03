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

## 1. Getting started : 

This workshop uses **pipenv** virtual environment for package management. 

To install run:
```
pip install pipenv

```
Then, clone the repository to your local session and install the virtual environment with: 

```
pipenv install 

```
Needed libraries are specified in the Pipfile. 

Finally declare instance of the virtual environment by runing: 

```
pipenv shell 

```


For more on Pipenv consider checking out the dedicated part in Visium's Onboarding project. 


### Requirements 

Make to sure to have 13 MB of memory available on disk for running the workshop.

### Run 

DVC pipeline will sequentially build the needed data sets, train the model, evaluate on test set and even plot training metrics. All parameters used for this execution can be found and changed in __dvc_workshop/params.py__.

To run the pipeline, the command : 
```
dvc repro

```
This triggers the following pipeline: 
##### Pipeline overview 
<pre>

      +--------------+         
      | data/raw.dvc |         
      +--------------+         
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

Each step corresponds to a stage in the Machine Learning workflow, as such: 
- preprocess: Filters out poster with red color in given range and does offline augmentation with rotation, scaling etc... than creates train, val, test split. Outputs processed images and splits under __data/preprocess__.
- train: Trains models on mutli-label classification task outputs trained model under __models/__ along side training history and finetuning history. 
- evaluate: Loads trained model from training steps and evaluates on test set produced during preprocessing. Performance and metrics are saved under __results/metrics__. 
- plot: Plots losses and accuracy during of the model during training and saves figures under __results/plots__.

## 2. DVC Remote

Disclaimer: All steps bellow have already been performed in the repository and are only being presented. 

Data, models, plots are version tracked with DVC and saved in a remote cloud storgae bucket. As such, versions are shared among developers and can be quickly recovered. Running: 
```
dvc pull
```
allows to get latest file versions form remote.

To set up the remote, one must first run a DVC instance by 
```
dvc init
```
This creates .dvc folder containing all cache files created by DVC for tracking.  

Then instentiate declare the remote, in our case with: 
```
dvc remote add -d dvc-workshop gs://dvc-workshop-cache/
```
This updates the __.dvc/.config__ file with the according information.

Now DVC knows where to store changes, but data is not traked yet.

One will see bellow that in DVC pipelines, outputs are automatically tracked but for now, lets focus on the raw data. 

Suppose one just copied locally the raw data from a remote storage (could be cloud storage bucket), to track the data, one runs: 

```
dvc add data/raw
```
This command generates hashes for each file, as snapshot of the file version. Later, theses hashes are sent to the remote to store them and make them available for any new developper starting to work on the project. 
To store hashes in the remote, one runs:

```
dvc push
```

## 3. DVC Pipeline

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
