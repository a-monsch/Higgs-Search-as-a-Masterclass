# Higgs-Search-as-a-Masterclass

## Brief overview
This repository contains a demo for [SMuK 2023](https://smuk23.dpg-tagungen.de/) with an advanced masterclass aimed for high school students with knowledge of particle physics from previous masterclasses. The goal is to perform a search for the Higgs boson on the $\mathrm{H}\rightarrow\mathrm{ZZ}\rightarrow 4\ell$ decay channel. For this purpose, a subset of CMS measurements taken in 2012 is used, which have been made available on the [CERN Open Data Portal](http://opendata.cern.ch/record/5500) [[1]](#1). The goal is to promote advanced concepts such as algorithm development or demonstrate the importance of data preparation in addition to the actual search. Likewise, students learn to assess whether an observation is significant with the aim of an application of these skills also outside of particle physics.

The Masterclass uses the Jupyter Notebook and relies on the Python programming language and takes inspiration of the ROOT based approach described in [[1]](#1). Basic knowledge for editing the notebook should be brought along or learned in the course of editing together with a supervisor. In general, the processing of the notebook is based on an alternating processing of the notebook and the interactive development of the concepts through the dialogue with the supervisor.

## Language used
 - German

## Execution viability
It is possible to run the repo using
[MyBinder](www.mybinder.org).

* To run the repo remotely:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-monsch/Higgs-Search-as-a-Masterclass/HEAD)

* To run locally (following commands for Terminal/Power Shell):

  ```
  git clone https://github.com/a-monsch/Higgs-Search-as-a-Masterclass
  cd Higgs-Search-as-a-Masterclass
  ```
  With an optional virtual environment:
  ```
  virtualenv venv
  # Linux
  source venv/bin/activate
  # Windows
  .\venv\Scripts\activate
  ```
  The necessary Python (>= 3.6) packages are listed below (`pip install <package>`) but can also be
  downloaded automatically via
  ```
  pip3 install -r binder/requirements.txt
  ```
   - [NumPy](https://numpy.org/)
   - [Pandas](https://pandas.pydata.org/)
   - [matplotlib](https://matplotlib.org/)
   - [Jupyter](https://jupyter.org/)
   - [tqdm](https://github.com/tqdm/tqdm)
   - [vector](https://github.com/scikit-hep/vector)

  If the virtual environment is used, a kernel for the jupyter notebooks
  can be reregistered.

  ```
  ipython kernel install --user --name=venv
  ```

  The jupyteter notebook can be started directly from within the directory of the repository with
  ```
  jupyter notebook
  ```
  After shutting down the notebook, you can leave the virtual environment with `deactivate`.

## Provided datasets
In the MyBinder version the data sets have been downloaded automatically.
For the local version, the data records can either be downloaded
[here](https://www.dropbox.com/s/zgrvm1idl4y6pj3/data_for_higgs_search_masterclass.zip?dl=0) manually in the `data` direcotry or be
downloaded and unpacked automatically by `sh /binder/postBuild` (Linux).


## References
<a id="1">[1]</a>
Bin Anuar Afiq Aizuddin Jomhari Nur Zulaiha Geiser Achim. *Higgs-to-four-leptonanalysis example using 2011-2012 data*. 2017. DOI: [10.7483/OPENDATA.CMS.JKB8.D634](10.7483/OPENDATA.CMS.JKB8.D634). URL: [http://opendata.vern/record/5500](http://opendata.vern/record/5500).
