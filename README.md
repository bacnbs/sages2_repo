# EEGFeaturesExtraction
This is a repository for
1. extracting and storing features from resting state EEG data.
2. running data visualization and statistical tests.
3. running machine learning tasks.

This README.md is for general usage/getting start. For task/project specific README, go to wikis folder.

Please email me if you're having trouble setting up or you found any bugs at mning@bidmc.harvard.edu. (If I'm no longer employed at BIDMC, you can report bug at GitHub's Issues.)

## Getting Start
None requires admin rights:
1. [git](https://git-scm.com/downloads)
2. GitHub account
3. [miniforge](https://github.com/conda-forge/miniforge)
4. [python 3.10](https://www.python.org/downloads/windows/) (DO NOT RUN AS ADMIN RIGHTS)

Notice on Git installation or configuration if already installed:
If you're installing Git for the first time, on the set-up page, click on the <strong>Override the default branch name for new repositories</strong> and in the textbox, put down `main`.

If you've already installed Git, you can run the following command:
```
git config --global init.defaultBranch main
```
This will change the default branch name from master to main.

## Setting up Repository
Whether you're interested in making contribution to this repository or just running the codes for your own project, I strongly suggest GitHub forking and cloning this over cloning alone. Instructions for both forking and cloning are shown below.

### Option 1: GitHub fork and clone
Copied/modified from [scikit-learn](https://scikit-learn.org/dev/developers/contributing.html)
1. Create an account on GitHub if you do not already have one.
2. Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see this guide.
3. Clone your fork of the scikit-learn repo from your GitHub account to your local disk. Open miniforge's command prompt and run the following command::
```
git clone git@github.com:YourLogin/EEGFeaturesExtraction.git  # add --depth 1 if your connection is slow
cd scikit-learn
```
4. Follow steps 2-6 in [Building from source](https://scikit-learn.org/dev/developers/advanced_installation.html#install-bleeding-edge) to build scikit-learn in development mode and return to this document.
5. Create a new python environment and install the development dependencies:
   - For the full setup (can be slow), open miniforge's command prompt and run the following command:
     ```
     conda create --name mne --file requirements_conda.txt
     ```
   - For the minimal setup (faster), open miniforge's command prompt and run the following command (miniforge should've already install libmamba). You'll need to install additional packages as needed:
     ```
     conda create --solver=libmamba --override-channels --channel=conda-forge --name=mne mne
     ```
6. Add the upstream remote. This saves a reference to the main repository, which you can use to keep your repository synchronized with the latest changes. In miniforge's command prompt:
```
git remote add upstream git@github.com:NoPenguinsLand/EEGFeaturesExtraction.git
```
7. Check that the upstream and origin remote aliases are configured correctly by running git remote -v which should display. In miniforge's command prompt:
```
origin  git@github.com:YourLogin/EEGFeaturesExtraction.git (fetch)
origin  git@github.com:YourLogin/EEGFeaturesExtraction.git (push)
upstream        git@github.com:NoPenguinsLand/EEGFeaturesExtraction.git (fetch)
upstream        git@github.com:NoPenguinsLand/EEGFeaturesExtraction.git (push)
```

### Option 2: Git clone to local drive
To clone the repository into a local directory:
1. In Miniforge command prompt, go to the local directory where you want the cloned repo to reside. For example, if your local directory is <strong>C:\Users\this_user\Documents\GitHub_Repos</strong>, then run the following command: ```cd "C:\Users\this_user\Documents\GitHub_Repos"```
2. Then to clone only a single brnach of the remote repository to the local direcoty, run the following command: ```git clone -b SharedCopy --single-branch https://github.com/NoPenguinsLand/EEGFeaturesExtraction.git```.

It should looks sometihng like this:
```Cloning into 'EEGFeaturesExtraction'...
remote: Enumerating objects: 350, done.
remote: Counting objects: 100% (350/350), done.
remote: Compressing objects: 100% (234/234), done.
remote: Total 350 (delta 214), reused 240 (delta 108), pack-reused 0
Receiving objects: 100% (350/350), 963.94 KiB | 801.00 KiB/s, done.
Resolving deltas: 100% (214/214), done.
Updating files: 100% (53/53), done.
```

### Update local repo with the latest changes from the remote branch
To update the cloned repo on local directory, change directory to the location of local repo. For example, if your local directory is <strong>C:\Users\this_user\Documents\GitHub_Repos</strong>, then it can be done in one of the two ways:

1. Open command prompt and run the following command: ```cd "C:\Users\this_user\Documents\GitHub_Repos"```
2. In File Explorer, go to the aforementioned directory, right click and select the "Git Bash Here" option (if you have git installed)

Then run the following command:
```
git checkout main
git fetch upstream
git merge upstream/main
```

It should looks something like this:

```
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), 1011 bytes | 1024 bytes/s, done.
From https://github.com/NoPenguinsLand/EEGFeaturesExtraction
 * branch            SharedCopy -> FETCH_HEAD
   976ec6a..d1c9509  SharedCopy -> origin/SharedCopy
Updating 976ec6a..d1c9509
Fast-forward
 README.md | 8 +++-----
 1 file changed, 3 insertions(+), 5 deletions(-)
```

To update the clone on R drive, change directory to the following path:

<strong>
"R:\Studies_Supporting Information\Standardized Data Analysis\GitHub_Repos\EEGFeaturesExtraction"
</strong>

This can be done in one of two ways:
1. Open command prompt and run the following command: ```cd /d "R:\Studies_Supporting Information\Standardized Data Analysis\GitHub_Repos\EEGFeaturesExtraction"```
2. In File Explorer, go to the aforementioned directory, right click and select the "Git Bash Here" option (if you have git installed)

Then run the following command:
```
git checkout main
git fetch upstream
git merge upstream/main
```

### Create new local branch
Before making any changes to the codes, create a new local branch called `devBrnch` branch with the following command.

```git checkout -b devBrnch```

This is where I'd suggest making changes to the codes.

### Install pre-commit and pre-commit hooks
Pre-commit hooks are a great way to both identify coding issues and enforce best coding practices before committing changes to local repository.
In miniforge command prompt, activate `mne` environment first and run the following commands
```
pip install pre-commit
pre-commit install
```
pre-commit checks can be disabled for a particular commit with `git commit -n`.

## Code Development
### Add changes to local repository


```
git add .
pre-commit run
```
if `pre-commit run` returns with errors, it'll either automatically fix whenever possible or marked errors to be manually fixed. After fixing all errors from pre-commit hooks, add the files to index by running the following commands
```
git add .
```

### Push changes from local repository to remote repository
After you've added changes to the local repository, you can push these to the remote repository. Here, we'll be using forked repository as the remote repository.
```
git commit -m "message"
git push origin devBrnch
```

### To run code from command prompt
To run from command prompt:
1. open Miniforge Prompt
2. Run the following command: ```mamba activate mne```. This will use the mne virtual environment with the python interpreter and all of the required python packages.
3. Using Miniforge Prompt, change directory to the one above EEGFeaturesExtraction head directory. (E.g., your repo is in ..\CNBS_Projects\EEGFeaturesExtraction, cd to <strong>"C:\Users\mning\Documents\CNBS_Projects"</strong>.)
4. run ```python -m EEGFeaturesExtraction.<script_name>```

### To run code from IDE
I personally use [PyDev for Eclipse](https://www.pydev.org/manual_101_install.html).

### Contribute to this repository for the BA-CNBS organization
To make contributions, first please let me know so I can minimize potential merge conflicts.

Again, I strongly suggest using GitHub's Pull Request. [See next for differences between Pull Request and local merge](#pull-request-vs-local-merge)

### Pull Request vs Local Merge
For major changes, I'd suggest doing Pull Request on GitHub. This way, we can use GitHub's code review platform. For minor changes, merging local branches before pushing upstream is fine since the main branch isn't protected.

## Appendix
### Resources for learning Git
1. [Visualization of Git](https://www.ndpsoftware.com/git-cheatsheet.html#loc=index)
2. [The Bible of Git](https://git-scm.com/book/en/v2) there's no substitute.
