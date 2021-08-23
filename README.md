# A Hands-on Introduction to Automatic Differentiation

This is the code repository accompanying the two-parts blog posts series: 
- [A Hands-on Introduction to Automatic Differentiation - Part 1](https://mostafa-samir.github.io/auto-diff-pt1/).
- [Build Your own Deep Learning Framework - A Hands-on Introduction to Automatic Differentiation - Part 2](https://mostafa-samir.github.io/auto-diff-pt2/)

## How to use the code

Before using the code, there are some requirements that needs to be present on your machine to run the code samples in the Jupyter notebooks. Some of these requirements are internal and can be easily fetched with Python's `pip` across the different platforms. Other requirements are external and needs to be fetched from other sources than Python's `pip`, and these sources usually differ based on your operating system. But fear not, it's quite simple to get them and you'll find the how-to instruction below.

### Installing external requirements for the visualizations
For the code to work, we need to have both `graphviz` and `ffmpeg` to be installed on your machine. These packages are mainly concerned with the computational graph visualizations and the animated reverse automatic differentiation visualizations. We here provide the instructions on how to install them on Ubuntu, macOs, and Windows.

#### Installing `graphviz`
##### Ubuntu
* Simply run `sudo apt install graphviz`
##### macOS
1. Install [Homebrew](https://brew.sh/) package manager.
2. Run `brew install graphviz`
##### Windows
1. Download the _.msi_ installer from [graphviz website](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)
2. Run the installer.
3. Locate the install location (which will probably be `C:\Program Files (x86)\GraphViz2.38`)
4. Add `C:\Program Files (x86)\GraphViz2.38\bin` to your `PATH` environment variable.
#### Installing `ffmpeg`
##### Ubuntu
* Simply run `sudo apt install ffmpeg`
##### macOS
1. Install [Homebrew](https://brew.sh/) package manager.
2. Run `brew install ffmpeg`
##### Windows
1. Download the windows static build from [ffmpeg website](https://ffmpeg.zeranoe.com/builds/), which is a zip file.
2. Unzip the build file in your preferred location (Let it be `C:\Program Files (x86)\ffmpeg`)
4. Add `C:\Program Files (x86)\ffmpeg\bin` to your `PATH` environment variable.
### Installing python requirements and running the code

_this assumes that you have python3.5 installed on your machine and you know how to use Jupyter notebooks_

1. Make sure that you have the `virtualenv` package by running `pip3 install virtualenv`
2. Open the terminal at your local copy of this repository and create a fresh virtual environment with `python3 -m venv venv`
3. Activate your new virtual environment with `source venv/bin/activate`
4. Run `pip install -r requirements.txt`
5. Install an IPython notebook kernel pointing to your virtual environment to use with the notebooks via `python -m ipykernel install --user --name AD`
6. Fire up jupyter notebook with `jupyter notebook` and start using the code.