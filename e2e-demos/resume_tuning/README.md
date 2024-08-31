# Resume Tuning App

## Pre-requisites

* Install Python 3.11
* Create Virtual Environment
* Activate
* Install needed libraries

## Prepare input documents

1. Need to copy paste the job application in a md file, be sure to use the md headers as

    "#" = "Header 1"
    "##"= "Header 2"
    "###"= "Header 3"
    "####"= "Header 4"

2. Get resume in pdf or md format. The resume may have a lot of skills and projects listed, the system will take the most appropriate.

## Usage

python main.py -a application_file.md -r resume.pdf -c firstname