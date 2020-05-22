library(reticulate)
library(shiny)

data_path <- file.path(getwd(),"Data")

# Define any Python packages needed for the app here:
PYTHON_DEPENDENCIES = c("opencv-python","numpy","pandas","scipy","more_itertools","statistics")


# ------------------ App virtualenv setup (Do not edit) ------------------- #

virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
python_path = Sys.getenv('PYTHON_PATH')

# Create virtual env and install dependencies
reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES)
reticulate::use_virtualenv(virtualenv_dir, required = T)

user_add_del <- import_from_path("user_add_del")
single_user_add <- user_add_del$single_user_add
multi_user_add <- user_add_del$multi_user_add
rm_users <- user_add_del$rm_users
testing <- import_from_path("testing")$testing


source("ui.R")
sys.source("server.R",chdir = T)

shinyApp(ui = ui, server = server)