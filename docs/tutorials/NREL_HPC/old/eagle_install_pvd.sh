#!/bin/bash

# Check if an argument has been passed
if [ $# -eq 0 ]; then
  echo "No arguments provided. Please provide a folder name."
  exit 1
fi

# Check if the folder already exists.
if [ -d "$1" ]; then
  echo "Folder '$1' already exists."
  read -p "Do you want to delete the folder? (y/n): " answer

  if [ "$answer" == "y" ]; then
    rm -r $1
    echo "Folder deleted."
  elif [ "$answer" == "n" ]; then
    echo "Aborting the operation."
    exit 1
  else
    echo "Invalid input. Please enter either 'y' or 'n'."
    exit 1
  fi
fi

# Create folder
mkdir "$1"
 
# Confirm the folder has been created
if [ -d "$1" ]; then
  echo "New folder '$1' has been created."
else
  echo "Failed to create folder '$1'."
fi

# Load conda on eagle
module purge; module load conda
python --version

# Create conda environment with the first argument as the name
#Note: there is currently a bug in numpy/numba python 3.10
conda create --name $1 python=3.9 anaconda
conda activate $1

# Check if CONDA environment is activated.
CENV="$(basename "$CONDA_DEFAULT_ENV")"
if [ $CENV == $1 ]; then
  echo New environment "$CENV" activated!
else
  echo "Failed to activate conda environment!"
  exit 1
fi

# Install required packages
# Install pvlib
conda install -c pvlib pvlib

# Clone the necessary repositories
git clone https://github.com/NREL/pvdeg.git ./$1/pvd
git clone https://github.com/NREL/gaps.git ./$1/gaps
git clone https://github.com/NREL/reV.git ./$1/reV

# Install rev 
echo "Installing reV"
pip install -e ./$1/reV

# Install gaps
# Note: gaps requirements are for 3.10 because of the bug I use --no-deps for now.
echo "Installing gaps"
pip install --no-deps -e ./$1/gaps 

# Install pvd from the dev_gaps branch for CLI
echo "Installing pvd"
cd ./$1/pvd
git checkout dev_gaps
pip install -e .

# Add conda environment to Jupyterhub on Europa
echo "Adding conda environment to Europa"
ipython kernel install --user --name $1

# Check that everything worked
pvd