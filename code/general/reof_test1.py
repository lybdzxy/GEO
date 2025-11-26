import xarray as xr
import seaborn as sns
from xeofs.models import EOF, EOFRotator
from scipy import signal
sns.set_context('paper')

import warnings
warnings.filterwarnings('ignore')

ssta = xr.open_dataset('E:/Downloads/ssta.nc')['ssta']

eofs = []
pcs = []
# (1) Standard EOF without regularization
model = EOF(ssta, dim=['time'], weights='coslat')
model.solve()
eofs.append(model.eofs())
pcs.append(model.pcs())
# (2) Varimax-rotated EOF analysis
model_var = EOF(ssta, dim=['time'], weights='coslat')
model_var.solve()
rot_var = EOFRotator(model, n_rot=50, power=1)
eofs.append(rot_var.eofs())
pcs.append(rot_var.pcs())
# (3) Promax-rotated EOF analysis
model_pro = EOF(ssta, dim=['time'], weights='coslat')
model_pro.solve()
rot_pro = EOFRotator(model, n_rot=50, power=2)
eofs.append(rot_pro.eofs())
pcs.append(rot_pro.pcs())