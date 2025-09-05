# 3D Mesh editing with pyopengl
some implementations and experiments for mesh surface editing using python + opengl

## Setup
Experimented on Python 3.8.10

```bash
pip install -r requirements.txt
```

## List of implementations
1. ~~pca blendshape~~
2. Computational Caricaturization of Surfaces 
    - `ccs.ipynb` `[WIP]`
3. Expression Cloning, [Noh and Neumann 2011] (RBF not implemented yet)
    - `expression_cloning.ipynb`
4. Deformation Transfer for Triangle Meshes, [Sumner and Popović]
    - `deformation_transfer.ipynb`
5. Compressed Skinning for Facial Blendshapes
    - `compressed_skinning.ipynb`
6. Laplacian Surface Editing [Sorkine et al. 2004]
    - `laplacian_surface_editing.py`
    - `laplacian_coating_transfer.py` `[WIP]`
7. As-Rigid-As-Possible Surface Modeling [Sorkine and Alexa 2007]
    - `ARAP.py`
8. Mean Value Coordinates for Closed Triangular Meshes [Ju et al. 2005]
    - `mean_value_coordinate.ipynb`
9. Harmonic Coordinates for Character Articulation [Joshi et al. 2007]
    - `harmonic_coordinates.ipynb` `[WIP]`

## Toy notebooks
- `bbox.ipynb` toy notebook for bounding box
- `filter_test.ipynb` toy notebook for 1D filter


## TODO:
    - [ ] clean up code
    - [ ] must clean up code ...!
    - [ ] maybe some fluid simulations? idk
