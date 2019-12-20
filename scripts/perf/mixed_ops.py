from FastAutoAugment.nas.operations import MixedOp
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from timebudget import timebudget

timebudget.set_quiet(True)
timebudget.set_show_all_stats(True)

cudnn.enabled = True
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
cudnn.benchmark = True
torch.cuda.set_device(0)

device = torch.device('cuda')

mop = MixedOp(16,1).to(device=device)

a = torch.randn(8, requires_grad=True).to(device=device)
x = torch.randn((64,16,32,32), requires_grad=True).to(device=device)

for i in range(1000):
    y = mop(x, a)
timebudget.report()

"""
Without cudnn setup, requires_grad=False:
                        3:    0.90ms for   1000 calls [stddev:    9.08, min:    0.49, max:  287.68]
                        4:    0.57ms for   1000 calls [stddev:    0.16, min:    0.48, max:    3.89]
                        6:    0.32ms for   1000 calls [stddev:    0.07, min:    0.27, max:    1.22]
                        5:    0.32ms for   1000 calls [stddev:    0.06, min:    0.27, max:    0.56]
                        0:    0.29ms for   1000 calls [stddev:    0.09, min:    0.19, max:    1.87]
                        1:    0.19ms for   1000 calls [stddev:    0.05, min:    0.16, max:    1.19]
                        7:    0.09ms for   1000 calls [stddev:    0.02, min:    0.07, max:    0.16]
                        2:    0.05ms for   1000 calls [stddev:    0.01, min:    0.04, max:    0.11]

With cudnn setup, requires_grad=False:
                        3:    0.86ms for   1000 calls [stddev:    8.12, min:    0.53, max:  257.40]
                        4:    0.54ms for   1000 calls [stddev:    0.06, min:    0.49, max:    0.91]
                        0:    0.31ms for   1000 calls [stddev:    0.04, min:    0.22, max:    1.03]
                        5:    0.30ms for   1000 calls [stddev:    0.03, min:    0.27, max:    0.52]
                        6:    0.30ms for   1000 calls [stddev:    0.03, min:    0.27, max:    0.51]
                        1:    0.19ms for   1000 calls [stddev:    0.05, min:    0.17, max:    1.48]
                        7:    0.09ms for   1000 calls [stddev:    0.01, min:    0.08, max:    0.17]
                        2:    0.05ms for   1000 calls [stddev:    0.01, min:    0.04, max:    0.10]

With cudnn setup, requires_grad=True:
    torch.Size([64, 16, 32, 32])7:    0.10ms for   1000 calls [stddev:    0.03, min:    0.09, max:    0.51]
    torch.Size([64, 16, 32, 32])6:    0.31ms for   1000 calls [stddev:    0.06, min:    0.27, max:    1.88]
    torch.Size([64, 16, 32, 32])5:    0.31ms for   1000 calls [stddev:    0.07, min:    0.27, max:    2.16]
    torch.Size([64, 16, 32, 32])4:    0.55ms for   1000 calls [stddev:    0.08, min:    0.49, max:    1.38]
    torch.Size([64, 16, 32, 32])3:    0.86ms for   1000 calls [stddev:    9.35, min:    0.49, max:  296.30]
    torch.Size([64, 16, 32, 32])2:    0.06ms for   1000 calls [stddev:    0.01, min:    0.05, max:    0.21]
    torch.Size([64, 16, 32, 32])1:    0.20ms for   1000 calls [stddev:    0.06, min:    0.18, max:    1.70]
    torch.Size([64, 16, 32, 32])0:    0.23ms for   1000 calls [stddev:    0.05, min:    0.20, max:    1.40]
                    forward:    2.78ms for   1000 calls [stddev:    9.43, min:    2.22, max:  300.59]

# PT-DARTS
                  forward:    2.70ms for    560 calls [stddev:    0.39, min:    2.22, max:    4.47]
                        3:    0.62ms for    560 calls [stddev:    0.09, min:    0.52, max:    1.14]
                        4:    0.61ms for    560 calls [stddev:    0.09, min:    0.51, max:    1.08]
                        6:    0.33ms for    560 calls [stddev:    0.05, min:    0.28, max:    0.76]
                        5:    0.33ms for    560 calls [stddev:    0.05, min:    0.28, max:    0.91]
                        0:    0.24ms for    560 calls [stddev:    0.04, min:    0.20, max:    0.40]
                        1:    0.21ms for    560 calls [stddev:    0.03, min:    0.18, max:    0.37]
                        2:    0.12ms for    560 calls [stddev:    0.17, min:    0.04, max:    1.74]
                        7:    0.10ms for    560 calls [stddev:    0.02, min:    0.08, max:    0.36]

# PT_DARTS
                  forward:    3.22ms for    560 calls [stddev:    0.84, min:    2.36, max:    8.14]
12/13 12:51:38 AM | Train: [ 1/50] Step 001/390 Loss 2.321 Prec@(1,5) (12.5%, 60.9%)
timebudget report...
torch.Size([64, 64, 8, 8])7:    0.12ms for    170 calls [stddev:    0.02, min:    0.09, max:    0.19]
torch.Size([64, 64, 8, 8])6:    0.36ms for    170 calls [stddev:    0.08, min:    0.29, max:    0.62]
torch.Size([64, 64, 8, 8])5:    0.36ms for    170 calls [stddev:    0.07, min:    0.29, max:    0.60]
torch.Size([64, 64, 8, 8])4:    0.67ms for    170 calls [stddev:    0.18, min:    0.53, max:    2.23]
torch.Size([64, 64, 8, 8])3:    0.68ms for    170 calls [stddev:    0.13, min:    0.54, max:    1.07]
torch.Size([64, 64, 8, 8])2:    0.07ms for    170 calls [stddev:    0.02, min:    0.05, max:    0.12]
torch.Size([64, 64, 8, 8])1:    0.23ms for    170 calls [stddev:    0.05, min:    0.19, max:    0.39]
torch.Size([64, 64, 8, 8])0:    0.26ms for    170 calls [stddev:    0.05, min:    0.21, max:    0.45]
torch.Size([64, 64, 16, 16])7:    0.14ms for     40 calls [stddev:    0.03, min:    0.11, max:    0.21]
torch.Size([64, 64, 16, 16])6:    0.37ms for     40 calls [stddev:    0.07, min:    0.30, max:    0.57]
torch.Size([64, 64, 16, 16])5:    0.37ms for     40 calls [stddev:    0.08, min:    0.30, max:    0.56]
torch.Size([64, 64, 16, 16])4:    0.67ms for     40 calls [stddev:    0.15, min:    0.54, max:    1.14]
torch.Size([64, 64, 16, 16])3:    0.67ms for     40 calls [stddev:    0.15, min:    0.55, max:    1.18]
torch.Size([64, 64, 16, 16])2:    0.50ms for     40 calls [stddev:    0.10, min:    0.41, max:    0.82]
torch.Size([64, 64, 16, 16])1:    0.23ms for     40 calls [stddev:    0.04, min:    0.19, max:    0.37]
torch.Size([64, 64, 16, 16])0:    0.26ms for     40 calls [stddev:    0.05, min:    0.21, max:    0.41]
torch.Size([64, 32, 32, 32])7:    0.14ms for     40 calls [stddev:    0.03, min:    0.11, max:    0.27]
torch.Size([64, 32, 32, 32])6:    0.38ms for     40 calls [stddev:    0.09, min:    0.29, max:    0.64]
torch.Size([64, 32, 32, 32])5:    0.38ms for     40 calls [stddev:    0.09, min:    0.29, max:    0.65]
torch.Size([64, 32, 32, 32])4:    0.70ms for     40 calls [stddev:    0.16, min:    0.54, max:    1.19]
torch.Size([64, 32, 32, 32])3:    0.70ms for     40 calls [stddev:    0.16, min:    0.54, max:    1.19]
torch.Size([64, 32, 32, 32])2:    0.54ms for     40 calls [stddev:    0.13, min:    0.41, max:    0.91]
torch.Size([64, 32, 32, 32])1:    0.24ms for     40 calls [stddev:    0.05, min:    0.19, max:    0.40]
torch.Size([64, 32, 32, 32])0:    0.28ms for     40 calls [stddev:    0.09, min:    0.21, max:    0.70]
torch.Size([64, 32, 16, 16])7:    0.11ms for    170 calls [stddev:    0.02, min:    0.09, max:    0.18]
torch.Size([64, 32, 16, 16])6:    0.34ms for    170 calls [stddev:    0.05, min:    0.29, max:    0.53]
torch.Size([64, 32, 16, 16])5:    0.34ms for    170 calls [stddev:    0.05, min:    0.29, max:    0.51]
torch.Size([64, 32, 16, 16])4:    0.63ms for    170 calls [stddev:    0.08, min:    0.54, max:    0.94]
torch.Size([64, 32, 16, 16])3:    0.64ms for    170 calls [stddev:    0.09, min:    0.54, max:    0.96]
torch.Size([64, 32, 16, 16])2:    0.06ms for    170 calls [stddev:    0.01, min:    0.05, max:    0.10]
torch.Size([64, 32, 16, 16])1:    0.22ms for    170 calls [stddev:    0.03, min:    0.19, max:    0.32]
torch.Size([64, 32, 16, 16])0:    0.25ms for    170 calls [stddev:    0.04, min:    0.21, max:    0.39]
torch.Size([64, 16, 32, 32])7:    0.13ms for    140 calls [stddev:    0.02, min:    0.09, max:    0.19]
torch.Size([64, 16, 32, 32])6:    0.39ms for    140 calls [stddev:    0.07, min:    0.30, max:    0.60]
torch.Size([64, 16, 32, 32])5:    0.39ms for    140 calls [stddev:    0.07, min:    0.30, max:    0.60]
torch.Size([64, 16, 32, 32])4:    0.72ms for    140 calls [stddev:    0.13, min:    0.55, max:    1.01]
torch.Size([64, 16, 32, 32])3:    0.74ms for    140 calls [stddev:    0.13, min:    0.56, max:    1.05]
torch.Size([64, 16, 32, 32])2:    0.07ms for    140 calls [stddev:    0.01, min:    0.06, max:    0.12]
torch.Size([64, 16, 32, 32])1:    0.25ms for    140 calls [stddev:    0.05, min:    0.19, max:    0.39]
torch.Size([64, 16, 32, 32])0:    0.29ms for    140 calls [stddev:    0.05, min:    0.22, max:    0.46]

# CPU
torch.Size([64, 16, 32, 32])7TrueTrue:    1.40ms for   1000 calls [stddev:    0.47, min:    0.60, max:    6.45]
torch.Size([64, 16, 32, 32])6TrueTrue:   62.79ms for   1000 calls [stddev:   15.29, min:   46.34, max:   97.40]
torch.Size([64, 16, 32, 32])5TrueTrue:   29.08ms for   1000 calls [stddev:    6.35, min:   21.32, max:   48.38]
torch.Size([64, 16, 32, 32])4TrueTrue:   17.52ms for   1000 calls [stddev:    1.53, min:   15.35, max:   29.74]
torch.Size([64, 16, 32, 32])3TrueTrue:   11.31ms for   1000 calls [stddev:    1.66, min:    9.30, max:   38.80]
torch.Size([64, 16, 32, 32])2TrueTrue:    0.63ms for   1000 calls [stddev:    0.29, min:    0.28, max:    4.34]
torch.Size([64, 16, 32, 32])1TrueTrue:    5.23ms for   1000 calls [stddev:    0.72, min:    3.32, max:   17.07]
torch.Size([64, 16, 32, 32])0TrueTrue:    7.89ms for   1000 calls [stddev:    0.91, min:    5.85, max:   21.50]
                  forward:  136.22ms for   1000 calls [stddev:   22.90, min:  109.06, max:  205.25]
"""