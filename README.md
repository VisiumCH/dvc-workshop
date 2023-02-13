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

## 1. Getting started : Cookiecutter Architecture 
 
The cookiecutter is a tool allowing to start Visium ML projects from a common template and a common tech stack.

It facilitates the sharing of common coding practices among engineers in Visium. The goal of the cookiecutter is especially to:

#### Foster collaboration between Visiumees
- Having a common tech stack makes Visumees be confronted to the same issues which makes it possible to support each other.
- Also sharing the same code template allows Visiumees to contribute to our development stack.

#### Ease the onboarding of new Visiumees
- Newcomers are directly taught the way everyone code here
- Avoid having to learn new coding practices every time we change project and team.

#### Facilitate project handover
- Switching engineers within projects is easier when all Visiumees share the same guidelines and project structure.

Repository architecture follows Visium's recommendation and reflects the company opiniotaded way of coding. 

As such, we are going to use this architecture for setting up the code architecture in this workshop.

### Instalation 

First, you will need to have cookiecutter package installed locally. If you don't have it already, you can set it up with __pip__ : 
```
pip install coockiecutter

```
Then start setting up the repository by running: 


```
cookiecutter dvc-workshop
```

Cookiecutter will ask you to enter

- project_name : __dvc_workshop__ 
