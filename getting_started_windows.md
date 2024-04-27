# Getting started from scratch on Windows

If you've got a Windows computer and nothing else, let's walk through how to get things set up so that you can run the homework assignments.

Many of these steps will be optional (there are other ways to do the same thing with different methods), but this is my preferred way to do things.

# Windows Instructions

## Install VSCode

VSCode is a really nice code editor which most people at Stanford and industry use. It also integrates well with WSL, which we'll install in a few steps.

See https://code.visualstudio.com/ and https://code.visualstudio.com/docs/remote/wsl for details. Select the Windows x64 download and follow the installer process

You'll need the following extensions to enable this with WSL:
- WSL https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl
- Remote Development https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack

You'll also need these for Python support:
- Python https://marketplace.visualstudio.com/items?itemName=ms-python.python
- Jupyter https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter

If you're interested in other helpful extensions, let me know. These are just the key ones

## Install Windows Subsystem for Linux (WSL)

It's possible to install Python, virtual environments, and all of the dependencies on Windows, but it's just a bit easier to do things on Linux (and it's a good thing to know how to do things in Linux for your job or research).

There are two options for using Linux (Ubuntu) with your Windows computer:
- Dual-booting
- Windows Subsystem for Linux

Dual-booting will be more performant: all of the RAM and CPU will be devoted to Ubuntu, since you'll only be running one operating system at a time. However, this also means that any time you want to use Ubuntu, you'll need to power off your Windows partition and log in to Ubuntu. This is a pain, especially if you need to frequently use software across both Windows and Linux

So, I recommend WSL: You can essentially run Windows and Linux at the same time, which is awesome. There are a few notable downsides though:
- Less available CPU and RAM (you're running two operating systems at the same time, so you have less resources available for Ubuntu)
- GUI applications are a lot slower to work with

You don't really need to worry about these downsides for anything in AA203 though.

To install, see https://learn.microsoft.com/en-us/windows/wsl/install for details. In short, all you need to do is:

- Search for Powershell in the Start menu
- Right click on Powershell and select "Run as administrator"
- Type `wsl --install` and hit Enter
- Reboot

*But what about the Windows store app? (Ubuntu for Windows)*: This is another way of enabling WSL that will also work. Personally, I don't like the Windows store as it is painfully slow on my laptop.

## Install Windows Terminal

This is the best way of interfacing with the Ubuntu command line. It should be already pre-installed on Windows 11, but if you're on Windows 10, you'll need to download it here https://apps.microsoft.com/detail/9n0dx20hk701?rtc=1&hl=en-us&gl=US

# WSL instructions

OK, now we have WSL installed and a terminal to access it from. You should be able to open a new tab in Windows Terminal and click the dropdown to select Ubuntu. 

Alternatively, if this does not work, you can access WSL by opening Powershell and typing `wsl`.

## Python setup

For best practice, we should be working with Python inside of a virtual environment. I like to use `pyenv` since it is fairly straightforward to use after the install process. `conda` can be quite annoying sometimes.

### Pyenv

To install `pyenv`, 

```
curl https://pyenv.run | bash
```

Then, set up your `~/.bashrc` -- Append the following lines to the bottom, using your preferred editor. If you're not comfortable editing this in the terminal (using `vim` for instance), you can type `explorer.exe .` and a Windows File Explorer window will open up. You can then click on the `.bashrc` file and edit it in something like Notepad.

```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

If the Pyenv Python install fails and warns about things not being installed, run this command (in WSL) to make sure the dependencies are up to date. (Then, retry the pyenv installation)

```
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Cool, now we have `pyenv` installed. Let's create a virtual environment for the class. I am using Python 3.10.8, which I know works for the homeworks.

Since this is probably the first time installing python, run 
```
pyenv install 3.10.8
```
Then,

```
pyenv virtualenv 3.10.8 aa203
pyenv shell aa203
```
The first line creates a virtual environment (`aa203`) with Python 3.10.8, and then the second line activates the virtual environment for the current terminal (shell) session.

## Cloning the course repo

I am going to install this in the home directory, but feel free to put this somewhere else.

```
cd ~
git clone https://github.com/StanfordASL/AA203-Homework
```

You should now have a folder called `AA203-Homework`. Let's navigate into it and use a `pyenv` trick to automatically activate your environment whenever you're in this repo. 

```
cd ~/AA203-Homework
pyenv local aa203
```

Also, we will regularly add new code to this repo, as new homeworks are released. Remember to `git pull` before any new assignment to retrieve the updates!

## Installing dependencies

You'll need a few key Python packages for the homework. We can automatically install these into your virtual environment by running the following:

```
cd ~/AA203-Homework
pip install -r requirements.txt
```

## Opening the homework in VSCode

You should be able to easily open the homework now. In your WSL homework directory (`~/AA203-Homework`), just type `code .` to open VSCode in this folder.

If this doesn't work, there's a chance that VSCode wasn't added to your `PATH` on installation. See the VSCode documentation here https://code.visualstudio.com/docs/editor/command-line

> Windows and Linux installations should add the VS Code binaries location to your system path. If this isn't the case, you can manually add the location to the Path environment variable ($PATH on Linux). For example, on Windows, the default VS Code binaries location is AppData\Local\Programs\Microsoft VS Code\bin. To review platform-specific setup instructions, see Setup.

Also, check out the following link to see how to add a directory to your path (https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho)

## Running the homework

You can run the homework scripts directly from the command line, or through VSCode. An example of running this from the command line would look like:

```
python ~/AA203-Homework/hw2/cartpole_balance.py
```

You can also run this in VSCode with the "Run" button at the top right. It should look like a triangle. Prior to doing this, you'll need to select a python interpreter to use in VSCode (https://code.visualstudio.com/docs/python/environments)

The quickest way to do this is:
- Press `ctrl` + `shift` + `p` at the same time to open the Command Palette
- Type `Python: Select Interpreter` and select it
- You should be able to see an option that looks like `Python 3.10.8 64-bit ('aa203')` -- Click on this to enable this environment in this VSCode window

# Other tips

VSCode tutorials: https://code.visualstudio.com/docs/getstarted/introvideos -- These are great, not too long, and cover all of the key concepts that will help you in the class and after the class.

Most of our homework code is just in python scripts, but we do provide a few Jupyter notebooks for recitations. Here is some info on how to set these up to run in VSCode: https://code.visualstudio.com/docs/datascience/jupyter-notebooks

# Concluding notes

Let me know (on Ed discussion) if you have any additional problems with the setup process! I'll update these instructions with FAQ as they arise. 
