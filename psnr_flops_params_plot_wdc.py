import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

FONT_SIZE = 8




y= [41.267, 36.756, 40.428, 34.552, 37.884, 39.695, 38.277, 38.444, 40.403, 41.375] #PSNR mixture noise
x = [23.21, 10.40, 33.37, 2.92, 96.81, 39.30, 14.97, 33.48, 23.51, 27.26] #flops
z = [0.687, 4.602, 0.662, 0.36, 14.28, 0.86, 1.91, 4.14, 0.52, 0.757] #params
model = ['HSDSSM++(Ours)', 'HCANet', 'TRQ3DNet', 'HSID-CNN', 'GRUNET', 'QRNN3D', 'SERT', 'SST', 'HSDT', 'HSDSSM']
marker = []
area = [i * 500 for i in z]

# Try to use Times New Roman, fallback to available serif font
try:
    # Check if Times New Roman is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' in available_fonts:
        hfont = {'fontname': 'Times New Roman', 'fontsize': FONT_SIZE}
    else:
        # Try common serif fonts available on Linux
        serif_fonts = ['DejaVu Serif', 'Liberation Serif', 'serif']
        font_found = False
        for font in serif_fonts:
            if font in available_fonts or font == 'serif':
                hfont = {'fontname': font, 'fontsize': FONT_SIZE}
                font_found = True
                break
        if not font_found:
            hfont = {'fontsize': FONT_SIZE}  # Use default font
except:
    hfont = {'fontsize': FONT_SIZE}  # Use default font if font manager fails

colors = ['green', 'darkblue','grey', 'cyan', 'purple','orange', 'magenta', 'blue', 'red', 'red', 'red', 'red', 'lightgreen', 'yellowgreen', 'steelblue']
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xscale("log")
for i in range(len(model)):
    scatter= ax.scatter(x[i], y[i], s=area[i], c=colors[i], alpha=0.7, cmap='jet', label=model[i])
    # if i==2:
    #     ax.annotate(model[i], xy=(x[i] - 41, y[i]), c='black', **hfont)
    if i==0:
        ax.annotate(model[i],xy=(x[i]-5, y[i]-0.4), c='black', **hfont)
    else:
        ax.annotate(model[i], xy=(x[i], y[i]), c='black', **hfont)
# ax.plot(x[8:12], y[8:12], marker='.', c='red', linestyle='dashed', alpha=0.4)
ax.set_xlabel('#FLOPs (G) Log-scale', **hfont)
ax.set_ylabel('MPSNR (dB)', **hfont)
ax.set_xlim(1.5, 155)
plt.grid(True, linestyle='--', which="both")

plt.tight_layout()
plt.savefig('psnr_flops_params_plot_wdc.pdf', dpi=300)
plt.show()