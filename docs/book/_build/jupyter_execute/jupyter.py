# Jupyter

JupyterLab is a library for interactive scientific computing in your web browser.

To open JupyterLab:
```bash
jupyter lab

# can specify a browser and port number
jupyter lab --browser=chrome --port=1234
```

Then open your browser and go to the session: `http://localhost:1234/`

Lots of [keyboard shortcuts](https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts) e.g., `a` to add cell above, `b` to add cell below, `dd` to delete cell, `shift + enter` to run cell, etc.

To add your conda environment for use in JupyterLab:
```bash
# activate the conda environment in the terminal
conda activate python3_teaching

# then add the ipython kernel
python -m ipykernel install --user --name python3_teaching --display-name "python3_teaching"
```

For more information, see the [documentation](https://jupyter.org/).
- [The JupyterLab Interface](https://jupyterlab.readthedocs.io/en/stable/user/interface.html).  
- [Working with Files](https://jupyterlab.readthedocs.io/en/stable/user/files.html).  
- [The Text Editor](https://jupyterlab.readthedocs.io/en/stable/user/file_editor.html).  
- [Notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html).  
- [Terminals](https://jupyterlab.readthedocs.io/en/stable/user/terminal.html).  
- [Managing Kernels and Terminals](https://jupyterlab.readthedocs.io/en/stable/user/running.html).  

