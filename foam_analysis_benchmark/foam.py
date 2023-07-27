
import importlib
import os

from multiprocessing import Pipe
from random import shuffle
from shutil import copy

from meow_base.conductors import LocalPythonConductor
from meow_base.core.runner import MeowRunner
from meow_base.functionality.file_io import make_dir, write_file, lines_to_string
from meow_base.functionality.requirements import create_python_requirements
from meow_base.patterns.file_event_pattern import WatchdogMonitor, \
    FileEventPattern
from meow_base.recipes.jupyter_notebook_recipe import PapermillHandler, \
    JupyterNotebookRecipe

TEST_MONITOR_BASE = "test_monitor_base"
TEST_DATA = "test_data"
POROSITY_CHECK_NOTEBOOK = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variables that will be overwritten accoring to pattern:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    f"input_filename = 'foam_ct_data{os.path.sep}foam_016_ideal_CT.npy'\n",
    "output_filedir_accepted = 'foam_ct_data_accepted' \n",
    "output_filedir_discarded = 'foam_ct_data_discarded'\n",
    "porosity_lower_threshold = 0.8\n",
    "utils_path = 'idmc_utils_module.py'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "import importlib\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "\n",
    "import importlib.util\n",
    "spec = importlib.util.spec_from_file_location(\"utils\", utils_path)\n",
    "utils = importlib.util.module_from_spec(spec)\n",
    "spec.loader.exec_module(utils)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 10000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Load data\n",
    "ct_data = np.load(input_filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "utils.plot_center_slices(ct_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_inds=np.random.randint(0, len(ct_data.ravel()), n_samples)\n",
    "n_components=2\n",
    "#Perform GMM fitting on samples from dataset\n",
    "means, stds, weights = utils.perform_GMM_np(\n",
    "    ct_data.ravel()[sample_inds], \n",
    "    n_components, \n",
    "    plot=True, \n",
    "    title='GMM fitted to '+str(n_samples)+' of '\n",
    "    +str(len(ct_data.ravel()))+' datapoints')\n",
    "print('weights: ', weights)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Classify data as 'accepted' or 'dircarded' according to porosity level\n",
    "\n",
    "Text file named according to the dataset will be stored in appropriate directories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    f"filename_withouth_npy=input_filename.split('{os.path.sep}')[-1].split('.')[0]\n",
    "\n",
    "if np.max(weights)>porosity_lower_threshold:\n",
    "    os.makedirs(output_filedir_accepted, exist_ok=True)\n",
    "    acc_path = os.path.join(output_filedir_accepted, \n",
    "                            filename_withouth_npy+'.txt')\n",
    "    with open(acc_path, 'w') as file:\n",
    "        file.write(str(np.max(weights))+' '+str(np.min(weights)))\n",
    "else:\n",
    "    os.makedirs(output_filedir_discarded, exist_ok=True)\n",
    "    dis_path = os.path.join(output_filedir_discarded, \n",
    "                            filename_withouth_npy+'.txt') \n",
    "    with open(dis_path, 'w') as file:\n",
    "        file.write(str(np.max(weights))+' '+str(np.min(weights)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
SEGMENT_FOAM_NOTEBOOK = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variables that will be overwritten accoring to pattern:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    f"input_filename = 'foam_ct_data_accepted{os.path.sep}foam_016_ideal_CT.txt'\n",
    "input_filedir = 'foam_ct_data'\n",
    "output_filedir = 'foam_ct_data_segmented'\n",
    "utils_path = 'idmc_utils_module.py'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import importlib\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import scipy.ndimage as snd\n",
    "import skimage\n",
    "\n",
    "import importlib.util\n",
    "spec = importlib.util.spec_from_file_location(\"utils\", utils_path)\n",
    "utils = importlib.util.module_from_spec(spec)\n",
    "spec.loader.exec_module(utils)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Segmentation\n",
    "\n",
    "Segmentation method used:\n",
    "\n",
    "- Median filter applied to reduce noise\n",
    "- Otsu thresholding applied to get binary data\n",
    "- Morphological closing performed to remove remaining single-voxel noise\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# importlib.reload(utils)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "median_filter_kernel_size = 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename_withouth_txt=input_filename.split(os.path.sep)[-1].split('.')[0]\n",
    "input_data = os.path.join(input_filedir, filename_withouth_txt+'.npy')\n",
    "\n",
    "ct_data = np.load(input_data)\n",
    "utils.plot_center_slices(ct_data, title = filename_withouth_txt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Median filtering "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_filtered = snd.median_filter(ct_data, median_filter_kernel_size)\n",
    "utils.plot_center_slices(data_filtered, title = filename_withouth_txt+' median filtered')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Otsu thresholding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "threshold = skimage.filters.threshold_otsu(data_filtered)\n",
    "data_thresholded = (data_filtered>threshold)*1\n",
    "utils.plot_center_slices(data_thresholded, title = filename_withouth_txt+' Otsu thresholded')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Morphological closing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_segmented = (skimage.morphology.binary_closing((data_thresholded==0))==0)\n",
    "utils.plot_center_slices(data_segmented, title = filename_withouth_txt+' Otsu thresholded')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Save data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename_save = filename_withouth_txt+'_segmented.npy'\n",
    "os.makedirs(output_filedir, exist_ok=True)\n",
    "np.save(os.path.join(output_filedir, filename_save), data_segmented)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
FOAM_PORE_ANALYSIS_NOTEBOOK = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variables that will be overwritten accoring to pattern:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    f"input_filename = 'foam_ct_data_segmented{os.path.sep}foam_016_ideal_CT_segmented.npy'\n",
    "output_filedir = 'foam_ct_data_pore_analysis'\n",
    "utils_path = 'idmc_utils_module.py'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import importlib\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import scipy.ndimage as snd\n",
    "\n",
    "from skimage.segmentation import watershed\n",
    "from skimage.feature import peak_local_max\n",
    "from matplotlib import cm\n",
    "from matplotlib.colors import ListedColormap, LinearSegmentedColormap\n",
    "\n",
    "import importlib.util\n",
    "spec = importlib.util.spec_from_file_location(\"utils\", utils_path)\n",
    "utils = importlib.util.module_from_spec(spec)\n",
    "spec.loader.exec_module(utils)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# importlib.reload(utils)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Foam pore analysis\n",
    "\n",
    "- Use Watershed algorithm to separate pores\n",
    "- Plot statistics\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = np.load(input_filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "utils.plot_center_slices(data, title = input_filename)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Watershed: Identify separate pores "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "#distance map\n",
    "distance = snd.distance_transform_edt((data==0))\n",
    "\n",
    "#get watershed seeds\n",
    "local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3, 3)), labels=(data==0))\n",
    "markers = snd.label(local_maxi)[0]\n",
    "\n",
    "#perform watershed pore seapration\n",
    "labels = watershed(-distance, markers, mask=(data==0))\n",
    "\n",
    "## Pore color mad\n",
    "somecmap = cm.get_cmap('magma', 256)\n",
    "cvals=np.random.uniform(0, 1, len(np.unique(labels)))\n",
    "newcmp = ListedColormap(somecmap(cvals))\n",
    "\n",
    "\n",
    "utils.plot_center_slices(-distance, cmap=plt.cm.gray, title='Distances')\n",
    "utils.plot_center_slices(labels, cmap=newcmp, title='Separated pores')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plot statistics: pore radii"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "volumes = np.array([np.sum(labels==label) for label in np.unique(labels)])\n",
    "volumes.sort()\n",
    "#ignore two largest labels (background and matrix)\n",
    "radii = (volumes[:-2]*3/(4*np.pi))**(1/3) #find radii, assuming spherical pores\n",
    "_=plt.hist(radii, bins=200)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Save plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename_withouth_npy=input_filename.split(os.path.sep)[-1].split('.')[0]\n",
    "filename_save = filename_withouth_npy+'_statistics.png'\n",
    "\n",
    "fig, ax = plt.subplots(1,3, figsize=(15,4))\n",
    "ax[0].imshow(labels[:,:,np.shape(labels)[2]//2], cmap=newcmp)\n",
    "ax[1].imshow(labels[:,np.shape(labels)[2]//2,:], cmap=newcmp)\n",
    "_=ax[2].hist(radii, bins=200)\n",
    "ax[2].set_title('Foam pore radii')\n",
    "\n",
    "os.makedirs(output_filedir, exist_ok=True)\n",
    "plt.savefig(os.path.join(output_filedir, filename_save))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
GENERATOR_NOTEBOOK = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# importing the necessary modules\n",
    "import numpy as np\n",
    "import random\n",
    "import os\n",
    "import shutil\n",
    "import importlib.util"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Variables to be overridden\n",
    "dest_dir = 'foam_ct_data'\n",
    "discarded = os.path.join('discarded', 'foam_data_0-big-.npy')\n",
    "utils_path = 'idmc_utils_module.py'\n",
    "gen_path = 'shared.py'\n",
    "test_data = 'test_data'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import loaded modules\n",
    "u_spec = importlib.util.spec_from_file_location(\"utils\", utils_path)\n",
    "utils = importlib.util.module_from_spec(u_spec)\n",
    "u_spec.loader.exec_module(utils)\n",
    "\n",
    "g_spec = importlib.util.spec_from_file_location(\"gen\", gen_path)\n",
    "gen = importlib.util.module_from_spec(g_spec)\n",
    "g_spec.loader.exec_module(gen)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Other variables, will be kept constant\n",
    "_, _, i, val, vx, vy, vz = os.path.basename(discarded).split('_')\n",
    "vz.replace(\".npy\", \"\")\n",
    "i = int(i)\n",
    "val = int(val)\n",
    "vx = int(vx)\n",
    "vy = int(vy)\n",
    "vz = int(vz)\n",
    "res=3/vz\n",
    "\n",
    "chance_good=1\n",
    "chance_small=0\n",
    "chance_big=0\n",
    "\n",
    "nspheres_per_unit_few=100\n",
    "nspheres_per_unit_ideal=1000\n",
    "nspheres_per_unit_many=10000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "possible_selection = [nspheres_per_unit_ideal] * chance_good \\\n",
    "    + [nspheres_per_unit_few] * chance_big \\\n",
    "    + [nspheres_per_unit_many] * chance_small\n",
    "random.shuffle(possible_selection)\n",
    "selection = possible_selection[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = f\"foam_dataset_{i}_{selection}_{vx}_{vy}_{vz}.npy\"\n",
    "backup_file = os.path.join(test_data, filename)\n",
    "if not os.path.exists(backup_file):\n",
    "    gen.create_foam_data_file(backup_file, selection, vx, vy, vz, res)\n",
    "target_file = os.path.join(dest_dir, filename)\n",
    "shutil.copy(backup_file, target_file)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
IDMC_UTILS_PYTHON_SCRIPT = [
    "import matplotlib.pyplot as plt",
    "from sklearn import mixture",
    "import numpy as np",
    "from skimage.morphology import convex_hull_image",
    "",
    "def saveplot(figpath_and_name, dataset):",
    "",
    "    fig, ax=plt.subplots(1, 3, figsize=(10, 4))",
    "    ax[0].imshow(dataset[dataset.shape[0]//2,:,:])",
    "    ax[1].imshow(dataset[:,dataset.shape[1]//2, :])",
    "    ax[2].imshow(dataset[:,:,dataset.shape[2]//2])",
    "    plt.savefig(figpath_and_name)",
    "",
    "",
    "def slice_by_slice_mask_calc(data):",
    "    '''calculate mask from convex hull of data, slice by slice in x-direction'''",
    "",
    "    mask=np.zeros(data.shape)",
    "    no_slices=data.shape[0]",
    "    for i in range(no_slices):",
    "        xslice=data[i,:,:]",
    "        mask[i,:,:]=convex_hull_image(xslice)",
    "    return mask",
    "",
    "",
    "def plot_center_slices(volume, title='', fig_external=[],figsize=(15,5), cmap='viridis', colorbar=False, vmin=None, vmax=None):",
    "        shape=np.shape(volume)",
    "",
    "        if len(fig_external)==0:",
    "            fig,ax = plt.subplots(1,3, figsize=figsize)",
    "        else:",
    "            fig = fig_external[0]",
    "            ax = fig_external[1]",
    "",
    "        fig.suptitle(title)",
    "        im=ax[0].imshow(volume[:,:, int(shape[2]/2)], cmap=cmap, vmin=vmin, vmax=vmax)",
    "        ax[0].set_title('Center z slice')",
    "        ax[1].imshow(volume[:,int(shape[1]/2),:], cmap=cmap, vmin=vmin, vmax=vmax)",
    "        ax[1].set_title('Center y slice')",
    "        ax[2].imshow(volume[int(shape[0]/2),:,:], cmap=cmap, vmin=vmin, vmax=vmax)",
    "        ax[2].set_title('Center x slice')",
    "",
    "        if colorbar:",
    "            fig.subplots_adjust(right=0.8)",
    "            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])",
    "            fig.colorbar(im, cax=cbar_ax)",
    "",
    "",
    "def perform_GMM_np(data_np, n_components, plot=False, n_init=1, nbins=500, title='', fig_external=[], return_labels=False):",
    "",
    "    #reshape data",
    "    n_samples=len(data_np)",
    "    X_train = np.concatenate([data_np.reshape((n_samples, 1)), np.zeros((n_samples, 1))], axis=1)",
    "",
    "    # fit a Gaussian Mixture Model",
    "    clf = mixture.GaussianMixture(n_components=n_components, covariance_type='full', n_init=n_init)",
    "    clf.fit(X_train)",
    "    if clf.converged_!=True:",
    "        print(' !! Did not converge! Converged: ',clf.converged_)",
    "",
    "    labels=clf.predict(X_train)",
    "",
    "    means=[]",
    "    stds=[]",
    "    weights=[]",
    "    for c in range(n_components):",
    "        component=X_train[labels==c][:,0]",
    "        means.append(np.mean(component))",
    "        stds.append(np.std(component))",
    "        weights.append(len(component)/len(data_np))",
    "",
    "    if plot:",
    "        gaussian = lambda x, mu, s, A: A*np.exp(-0.5*(x-mu)**2/s**2)/np.sqrt(2*np.pi*s**2)",
    "",
    "        if len(fig_external)>0:",
    "            fig, ax=fig_external[0], fig_external[1]",
    "        else:",
    "            fig, ax=plt.subplots(1, figsize=(16, 8))",
    "",
    "        hist, bin_edges = np.histogram(data_np, bins=nbins)",
    "        bin_size=np.diff(bin_edges)",
    "        bin_centers = bin_edges[:-1] +  bin_size/ 2",
    "        hist_normed = hist/(n_samples*bin_size) #normalizing to get 1 under graph",
    "        ax.bar(bin_centers,hist_normed, bin_size, alpha=0.5)",
    "        if len(title)>0:",
    "            ax.set_title(title)",
    "        else:",
    "            ax.set_title('Histogram, '+str(n_samples)+' datapoints. ')",
    "",
    "        #COLORMAP WITH EVENLY SPACED COLORS!",
    "        colors=plt.cm.rainbow(np.linspace(0,1,n_components+1))#rainbow, plasma, autumn, viridis...",
    "",
    "        x_vals=np.linspace(np.min(bin_edges), np.max(bin_edges), 500)",
    "",
    "        g_total=np.zeros_like(x_vals)",
    "        for c in range(n_components):",
    "            gc=gaussian(x_vals, means[c], stds[c], weights[c])",
    "            ax.plot(x_vals, gc, color=colors[c], linewidth=2, label='mean=%.2f'%(means[c]))",
    "            ax.arrow(means[c], weights[c], 0, 0.1)",
    "            g_total+=gc",
    "        ax.plot(x_vals, g_total, color=colors[-1], linewidth=2, label='Total Model')",
    "        plt.legend()",
    "",
    "    if return_labels:",
    "        return means, stds, weights, labels",
    "    else:",
    "        return means, stds, weights"
]
GENERATE_PYTHON_SCRIPT = [
    "import numpy as np",
    "import random",
    "import foam_ct_phantom",
    "",
    "def generate_foam(nspheres_per_unit, vx, vy, vz, res):",
    "    def maxsize_func(x, y, z):",
    "        return 0.2 - 0.1*np.abs(z)",
    "",
    "    random_seed=random.randint(0,4294967295)",
    "    foam_ct_phantom.FoamPhantom.generate('temp_phantom_info.h5',",
    "                                         random_seed,",
    "                                         nspheres_per_unit=nspheres_per_unit,",
    "                                         maxsize=maxsize_func)",
    "",
    "    geom = foam_ct_phantom.VolumeGeometry(vx, vy, vz, res)",
    "    phantom = foam_ct_phantom.FoamPhantom('temp_phantom_info.h5')",
    "    phantom.generate_volume('temp_phantom.h5', geom)",
    "    dataset = foam_ct_phantom.load_volume('temp_phantom.h5')",
    "",
    "    return dataset",
    "",
    "def create_foam_data_file(filename:str, val:int, vx:int, vy:int, vz:int, res:int):",
    "    dataset = generate_foam(val, vx, vy, vz, res)",
    "    np.save(filename, dataset)",
    "    del dataset"
]

pattern_check = FileEventPattern(
    "pattern_check", 
    os.path.join("foam_ct_data", "*"), 
    "recipe_check", 
    "input_filename",
    parameters={
        "output_filedir_accepted": 
            os.path.join("{BASE}", "foam_ct_data_accepted"),
        "output_filedir_discarded": 
            os.path.join("{BASE}", "foam_ct_data_discarded"),
        "porosity_lower_threshold": 0.8,
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })

pattern_segment = FileEventPattern(
    "pattern_segment",
    os.path.join("foam_ct_data_accepted", "*"),
    "recipe_segment",
    "input_filename",
    parameters={
        "output_filedir": os.path.join("{BASE}", "foam_ct_data_segmented"),
        "input_filedir": os.path.join("{BASE}", "foam_ct_data"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })

pattern_analysis = FileEventPattern(
    "pattern_analysis",
    os.path.join("foam_ct_data_segmented", "*"),
    "recipe_analysis",
    "input_filename",
    parameters={
        "output_filedir": os.path.join("{BASE}", "foam_ct_data_pore_analysis"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })


pattern_regenerate_random = FileEventPattern(
    "pattern_regenerate_random",
    os.path.join("foam_ct_data_discarded", "*"),
    "recipe_generator",
    "discarded",
    parameters={
        "dest_dir": os.path.join("{BASE}", "foam_ct_data"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py"),
        "gen_path": os.path.join("{BASE}", "generator.py"),
        "test_data": os.path.join(TEST_DATA, "foam_ct_data"),
        "vx": 32,
        "vy": 32,
        "vz": 32,
        "res": 3/32,
        "chance_good": 1,
        "chance_small": 0,
        "chance_big": 3
    })

patterns = {
    'pattern_check': pattern_check,
    'pattern_segment': pattern_segment,
    'pattern_analysis': pattern_analysis,
    'pattern_regenerate_random': pattern_regenerate_random
}

recipe_check_key, recipe_check_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib"])
recipe_check = JupyterNotebookRecipe(
    'recipe_check',
    POROSITY_CHECK_NOTEBOOK, 
    requirements={recipe_check_key: recipe_check_req}
)

recipe_segment_key, recipe_segment_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib", "scipy", "skimage"])
recipe_segment = JupyterNotebookRecipe(
    'recipe_segment',
    SEGMENT_FOAM_NOTEBOOK, 
    requirements={recipe_segment_key: recipe_segment_req}
)

recipe_analysis_key, recipe_analysis_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib", "scipy", "skimage"])
recipe_analysis = JupyterNotebookRecipe(
    'recipe_analysis',
    FOAM_PORE_ANALYSIS_NOTEBOOK, 
    requirements={recipe_analysis_key: recipe_analysis_req}
)

recipe_generator_key, recipe_generator_req = create_python_requirements(
    modules=["numpy", "matplotlib", "random"])
recipe_generator = JupyterNotebookRecipe(
    'recipe_generator',
    GENERATOR_NOTEBOOK, 
    requirements={recipe_generator_key: recipe_generator_req}           
)

recipes = {
    'recipe_check': recipe_check,
    'recipe_segment': recipe_segment,
    'recipe_analysis': recipe_analysis,
    'recipe_generator': recipe_generator
}

make_dir("job_output", ensure_clean=True)
make_dir("job_queue", ensure_clean=True)
make_dir(TEST_MONITOR_BASE, ensure_clean=True)

runner = MeowRunner(
    WatchdogMonitor(
        TEST_MONITOR_BASE,
        patterns,
        recipes,
        settletime=1
    ), 
    PapermillHandler(),
    LocalPythonConductor(pause_time=2)
)

# Intercept messages between the conductor and runner for testing
conductor_to_test_conductor, conductor_to_test_test = Pipe(duplex=True)
test_to_runner_runner, test_to_runner_test = Pipe(duplex=True)

runner.conductors[0].to_runner_job = conductor_to_test_conductor

for i in range(len(runner.job_connections)):
    _, obj = runner.job_connections[i]

    if obj == runner.conductors[0]:
        runner.job_connections[i] = (test_to_runner_runner, runner.job_connections[i][1])

good = 0
big = 20
small = 0
vx = 32
vy = 32
vz = 32
res = 3/vz
backup_data_dir = os.path.join(TEST_DATA, "foam_ct_data")
make_dir(backup_data_dir)
foam_data_dir = os.path.join(TEST_MONITOR_BASE, "foam_ct_data")
make_dir(foam_data_dir)

write_file(lines_to_string(IDMC_UTILS_PYTHON_SCRIPT), 
    os.path.join(TEST_MONITOR_BASE, "idmc_utils_module.py"))

gen_path = os.path.join(TEST_MONITOR_BASE, "generator.py")
write_file(lines_to_string(GENERATE_PYTHON_SCRIPT), gen_path)

all_data = [1000] * good + [100] * big + [10000] * small
shuffle(all_data)

u_spec = importlib.util.spec_from_file_location("gen", gen_path)
gen = importlib.util.module_from_spec(u_spec)
u_spec.loader.exec_module(gen)

for i, val in enumerate(all_data):
    filename = f"foam_dataset_{i}_{val}_{vx}_{vy}_{vz}.npy"
    backup_file = os.path.join(backup_data_dir, filename)
    if not os.path.exists(backup_file):
        gen.create_foam_data_file(backup_file, val, vx, vy, vz, res)

    target_file = os.path.join(foam_data_dir, filename)
    copy(backup_file, target_file)


print("starting runner")
runner.start()

loops = 0
idles = 0
while loops < 1200 and idles < 15:
    # Initial prompt
    if conductor_to_test_test.poll(60):
        msg = conductor_to_test_test.recv()
    else:
        break       
    test_to_runner_test.send(msg)

    # Reply
    if test_to_runner_test.poll(15):
        msg = test_to_runner_test.recv()
        if msg == 1:
            idles += 1
        else:
            idles = 0
    else:
        break      
    conductor_to_test_test.send(msg)

    loops += 1

print("stopping runner")
runner.stop()
