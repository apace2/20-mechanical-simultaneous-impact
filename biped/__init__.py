import pathlib

#from .Biped import Biped, PCrBiped, DecoupledBiped, RigidBiped

_data_folder = pathlib.Path(__file__).parent.absolute()
_fig_folder = pathlib.Path(__file__).parent.parent.absolute() / 'figs'

for folder in [_data_folder, _fig_folder]:
    folder.mkdir(parents=True, exist_ok=True)
